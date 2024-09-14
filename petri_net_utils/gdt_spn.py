from pm4py import PetriNet
from typing import Any, Collection, Dict


class GDTSPN(PetriNet):

    class ImmediateTransition(PetriNet.Transition):
        def __init__(self, name: str, label: str = None, in_arcs: Collection[PetriNet.Arc] = None, out_arcs: Collection[PetriNet.Arc] = None, weight: float = 1.0, properties: Dict[str, Any] = None):
            super().__init__(name, label, in_arcs, out_arcs, properties)
            self.__weight = weight

        def __set_weight(self, weight: float):
            self.__weight = weight

        def __get_weight(self) -> float:
            return self.__weight

        weight = property(__get_weight, __set_weight)


    class TimedTransition(PetriNet.Transition):
        def __init__(self, name: str, label: str = None, in_arcs: Collection[PetriNet.Arc] = None, out_arcs: Collection[PetriNet.Arc] = None, weight: float = 1.0, time_performance: Dict[str, float] = None,  properties: Dict[str, Any] = None):
            super().__init__(name, label, in_arcs, out_arcs, properties)
            self.__weight = weight
            self.__time_performance = time_performance

        def __set_time_performance(self, time_performance: Dict[str, float]):
            if time_performance['mean'] is not None:
                if time_performance['variance'] is not None:
                    self.__time_performance = time_performance

        def __get_time_performance(self) -> Dict[str, float]:
            return self.__time_performance

        def __set_weight(self, weight: float):
            self.__weight = weight

        def __get_weight(self) -> float:
            return self.__weight

        weight = property(__get_weight, __set_weight)
        time_performance = property(__get_time_performance, __set_time_performance)