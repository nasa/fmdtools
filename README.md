# Water Rescue fmdtools Simulation

This is a simulation of a beach water-rescue operation, built on the fmdtools resilience-modeling framework. It models a patrolling drone and a ground responder working together to detect and reach a distressed swimmer before their survival timer runs out. The drone flies a patrol path and drops a rescue buoy when reaching a distressed swimmer, while the responder sweeps a scan cone from its base and moves out to perform the rescue when either detecting a distresed swimmer itself or recieving an alert from the drone on patrol.

This is used to see how things will behave when something goes wrong, especially in scenarios that have never actually happened before. We referenced SysML models which were created from public data provided by news reports and the FDNY to model the behavior of the different actors in a scenario: the drone, the environment, and everything else involved in the scenario.

