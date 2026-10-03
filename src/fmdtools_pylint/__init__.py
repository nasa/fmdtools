# SPDX-License-Identifier: Apache-2.0
"""Opt-in static role inference for Pylint.

Use ``pylint --load-plugins=fmdtools_pylint model.py``. Model modules are parsed,
not imported or instantiated. Runtime simulation is not changed.
"""

from functools import partial

import astroid
from astroid import MANAGER, Uninferable, bases, inference_tip, nodes
from astroid.exceptions import AstroidError
from astroid.util import safe_infer
from pylint.checkers import BaseChecker
from pylint.checkers.utils import only_required_for_messages

_BASE_OBJECT = "fmdtools.define.object.base.BaseObject"
_BASE_CONTAINER = "fmdtools.define.container.base.BaseContainer"
_REGISTERED = False


def _infer_role(target, context=None, *, owner, declaration):
    """Resolve a declared class without calling its constructor."""
    try:
        values = list(owner.igetattr(declaration, context=context))
    except AstroidError:
        yield Uninferable
        return
    for value in values:
        yield (
            value.instantiate_class()
            if isinstance(value, nodes.ClassDef)
            else Uninferable
        )


def _transform_roles(node):
    """Expose generated roles as instance-valued attributes in Astroid only."""
    try:
        hierarchy = node.mro()
        if not any(cls.qname() == _BASE_OBJECT for cls in hierarchy):
            return
        roletypes = list(node.igetattr("roletypes"))
        if len(roletypes) != 1 or not isinstance(
            roletypes[0], (nodes.List, nodes.Tuple)
        ):
            return
        prefixes = []
        for role in roletypes[0].elts:
            if not isinstance(role, nodes.Const) or not isinstance(role.value, str):
                return
            prefixes.append(role.value + "_")
        declarations = {}
        for cls in reversed(hierarchy):
            for name in tuple(cls.locals):
                for prefix in prefixes:
                    if name.startswith(prefix) and name != prefix:
                        declarations[name[len(prefix) :]] = name
        for name, declaration in declarations.items():
            if any(
                name in cls.instance_attrs
                or any(
                    not getattr(value, "_fmdtools_role", False)
                    for value in cls.locals.get(name, ())
                )
                for cls in hierarchy
            ):
                continue
            assignment = astroid.parse(
                f"{name} = {declaration}()", apply_transforms=False
            ).body[0]
            assignment.parent = node
            target = assignment.targets[0]
            target._fmdtools_role = True
            inference_tip(partial(_infer_role, owner=node, declaration=declaration))(
                target
            )
            node.locals[name] = [target]
    except AstroidError:
        # Unresolvable bases or role declarations retain ordinary Pylint inference.
        return


def _known_container_bases(cls, seen=None):
    """Treat only BaseContainer's native base as an explicitly known boundary."""
    if cls.qname() == _BASE_CONTAINER:
        return True
    seen = set() if seen is None else seen
    if cls in seen:
        return False
    seen = seen | {cls}
    for base in cls.bases:
        inferred = safe_infer(base)
        if not isinstance(inferred, nodes.ClassDef):
            return False
        if not _known_container_bases(inferred, seen):
            return False
    return True


class ModelFieldsChecker(BaseChecker):
    """Check fixed-schema container fields without importing the C extension."""

    name = "fmdtools-model-fields"
    msgs = {
        "E9901": (
            "Container '%s' has no declared member '%s'",
            "fmdtools-unknown-field",
            "Used for an unknown member of a statically resolved fmdtools container.",
        ),
    }

    @only_required_for_messages("fmdtools-unknown-field")
    def visit_attribute(self, node):
        """Report only when every inferred container lacks the member."""
        if node.attrname.startswith("_"):
            return
        try:
            owners = list(node.expr.infer())
            missing = []
            for owner in owners:
                if not isinstance(owner, bases.Instance):
                    return
                hierarchy = [owner._proxied, *owner._proxied.ancestors()]
                if not any(cls.qname() == _BASE_CONTAINER for cls in hierarchy):
                    return
                if not _known_container_bases(owner._proxied):
                    return
                if owner.has_dynamic_getattr():
                    return
                if any(
                    isinstance(statement, nodes.AnnAssign)
                    and isinstance(statement.target, nodes.AssignName)
                    and statement.target.name == node.attrname
                    for cls in hierarchy
                    for statement in cls.body
                ):
                    return
                try:
                    owner.getattr(node.attrname)
                    return
                except astroid.AttributeInferenceError:
                    missing.append(owner.name)
            if missing:
                self.add_message(
                    "fmdtools-unknown-field",
                    node=node,
                    args=(", ".join(dict.fromkeys(missing)), node.attrname),
                )
        except AstroidError:
            return

    visit_assignattr = visit_attribute
    visit_delattr = visit_attribute


def register(linter):
    """Enable this opt-in transformation when loaded by Pylint."""
    global _REGISTERED
    if not _REGISTERED:
        MANAGER.register_transform(nodes.ClassDef, _transform_roles)
        _REGISTERED = True
    if linter is not None:
        linter.register_checker(ModelFieldsChecker(linter))
