import argparse
import sys
import threading
import time
from datetime import datetime

from pm4py import Marking
from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

from log_repair import log_repair
from petri_net_utils.gdt_spn import GDTSPN


def create_places(amount):
    p_map = {}
    name = "p_"
    for i in range(amount):
        place_name = name + str(i)
        place = GDTSPN.Place(place_name)
        p_map[i] = place
    return p_map


def create_gdtspn1():
    test_spn = GDTSPN("Test GDT_SPN")

    p_map = create_places(4)
    [test_spn.places.add(p) for p in p_map.values()]

    timing_data = {"mean": 4.0, "variance": 4.0}
    t1 = GDTSPN.TimedTransition("t1", "A", time_performance=timing_data)
    test_spn.transitions.add(t1)
    timing_data = {"mean": 7.0, "variance": 9.0}
    t2 = GDTSPN.TimedTransition("t2", "B", time_performance=timing_data)
    test_spn.transitions.add(t2)
    timing_data = {"mean": 11.0, "variance": 9.0}
    t3 = GDTSPN.TimedTransition("t3", "C", time_performance=timing_data)
    test_spn.transitions.add(t3)

    add_arc_from_to(p_map[0], t1, test_spn)
    add_arc_from_to(t1, p_map[1], test_spn)
    add_arc_from_to(p_map[1], t2, test_spn)
    add_arc_from_to(t2, p_map[2], test_spn)
    add_arc_from_to(p_map[2], t3, test_spn)
    add_arc_from_to(t3, p_map[3], test_spn)

    init_marking = Marking()
    init_marking[p_map[0]] = 1

    final_marking = Marking()
    final_marking[p_map[3]] = 1
    return test_spn, init_marking, final_marking


def create_gdtspn2():
    # Example from Rogge-Solti
    test_spn = GDTSPN("Test02 GDT_SPN")
    p_map = create_places(15)
    [test_spn.places.add(p) for p in p_map.values()]

    t1 = GDTSPN.ImmediateTransition("t1", "t1", None, None, 0.5)
    test_spn.transitions.add(t1)
    t2 = GDTSPN.ImmediateTransition("t2", "t2", None, None, 0.5)
    test_spn.transitions.add(t2)
    t3 = GDTSPN.ImmediateTransition("t3", "t3", None, None, 1)
    test_spn.transitions.add(t3)
    t4 = GDTSPN.ImmediateTransition("t4", "t4", None, None, 1)
    test_spn.transitions.add(t4)
    t5 = GDTSPN.ImmediateTransition("t5", "t5", None, None, 1)
    test_spn.transitions.add(t5)
    t6 = GDTSPN.ImmediateTransition("t6", "t6", None, None, 0.25)
    test_spn.transitions.add(t6)
    t7 = GDTSPN.ImmediateTransition("t7", "t7", None, None, 0.75)
    test_spn.transitions.add(t7)

    timing_data = {"mean": 20.0, "variance": 5.0 ** 2}
    ta = GDTSPN.TimedTransition("ta", "A", time_performance=timing_data)
    test_spn.transitions.add(ta)

    timing_data = {"mean": 16.0, "variance": 3.0 ** 2}
    tb = GDTSPN.TimedTransition("tb", "B", time_performance=timing_data)
    test_spn.transitions.add(tb)
    timing_data = {"mean": 9.0, "variance": 3.0 ** 2}
    tc = GDTSPN.TimedTransition("tc", "C", time_performance=timing_data)
    test_spn.transitions.add(tc)
    timing_data = {"mean": 10.0, "variance": 2.0 ** 2}
    td = GDTSPN.TimedTransition("td", "D", time_performance=timing_data)
    test_spn.transitions.add(td)
    timing_data = {"mean": 15.0, "variance": 4.0 ** 2}
    te = GDTSPN.TimedTransition("te", "E", time_performance=timing_data)
    test_spn.transitions.add(te)
    timing_data = {"mean": 11.0, "variance": 2.0 ** 2}
    tf = GDTSPN.TimedTransition("tf", "F", time_performance=timing_data)
    test_spn.transitions.add(tf)
    timing_data = {"mean": 10.0, "variance": 2.0 ** 2}
    tg = GDTSPN.TimedTransition("tg", "G", time_performance=timing_data)
    test_spn.transitions.add(tg)
    timing_data = {"mean": 5.0, "variance": 1.0 ** 2}
    th = GDTSPN.TimedTransition("th", "H", time_performance=timing_data)
    test_spn.transitions.add(th)

    add_arc_from_to(p_map[0], t1, test_spn)
    add_arc_from_to(t1, p_map[1], test_spn)
    add_arc_from_to(p_map[1], ta, test_spn)
    add_arc_from_to(ta, p_map[2], test_spn)
    add_arc_from_to(p_map[2], t3, test_spn)
    add_arc_from_to(t3, p_map[3], test_spn)
    add_arc_from_to(t3, p_map[4], test_spn)
    # to C and to D
    add_arc_from_to(p_map[3], tc, test_spn)
    add_arc_from_to(p_map[4], td, test_spn)
    add_arc_from_to(tc, p_map[5], test_spn)
    add_arc_from_to(td, p_map[6], test_spn)

    # lower path
    add_arc_from_to(p_map[0], t2, test_spn)
    add_arc_from_to(t2, p_map[7], test_spn)
    add_arc_from_to(p_map[7], tb, test_spn)
    add_arc_from_to(tb, p_map[8], test_spn)
    add_arc_from_to(p_map[8], t4, test_spn)
    add_arc_from_to(t4, p_map[9], test_spn)
    add_arc_from_to(t4, p_map[10], test_spn)

    # to E and to F
    add_arc_from_to(p_map[9], te, test_spn)
    add_arc_from_to(p_map[10], tf, test_spn)
    add_arc_from_to(te, p_map[5], test_spn)
    add_arc_from_to(tf, p_map[11], test_spn)
    add_arc_from_to(p_map[11], tg, test_spn)
    add_arc_from_to(tg, p_map[6], test_spn)

    add_arc_from_to(p_map[5], t5, test_spn)
    add_arc_from_to(p_map[6], t5, test_spn)

    # after join
    add_arc_from_to(t5, p_map[12], test_spn)
    add_arc_from_to(p_map[12], t6, test_spn)
    add_arc_from_to(t6, p_map[0], test_spn)

    add_arc_from_to(p_map[12], t7, test_spn)
    add_arc_from_to(t7, p_map[13], test_spn)
    add_arc_from_to(p_map[13], th, test_spn)
    add_arc_from_to(th, p_map[14], test_spn)

    init_marking = Marking()
    init_marking[p_map[0]] = 1

    final_marking = Marking()
    final_marking[p_map[14]] = 1

    return test_spn, init_marking, final_marking


def create_gdtspn3():
    # Example from Solti
    test_spn = GDTSPN("Test02 GDT_SPN with waiting time")
    p_map = create_places(23)
    [test_spn.places.add(p) for p in p_map.values()]

    t1 = GDTSPN.ImmediateTransition("t1", "t1", None, None, 0.5)
    test_spn.transitions.add(t1)
    t2 = GDTSPN.ImmediateTransition("t2", "t2", None, None, 0.5)
    test_spn.transitions.add(t2)
    t3 = GDTSPN.ImmediateTransition("t3", "t3", None, None, 1)
    test_spn.transitions.add(t3)
    t4 = GDTSPN.ImmediateTransition("t4", "t4", None, None, 1)
    test_spn.transitions.add(t4)
    t5 = GDTSPN.ImmediateTransition("t5", "t5", None, None, 1)
    test_spn.transitions.add(t5)
    t6 = GDTSPN.ImmediateTransition("t6", "t6", None, None, 0.25)
    test_spn.transitions.add(t6)
    t7 = GDTSPN.ImmediateTransition("t7", "t7", None, None, 0.75)
    test_spn.transitions.add(t7)

    timing_data = {"mean": 15.0, "variance": 4.0 ** 2}
    ta = GDTSPN.TimedTransition("ta", "A", time_performance=timing_data)
    test_spn.transitions.add(ta)
    timing_data = {"mean": 5.0, "variance": 3.0 ** 2}
    idlea = GDTSPN.TimedTransition("Idle_A", "Idle_A", time_performance=timing_data)
    test_spn.transitions.add(idlea)
    timing_data = {"mean": 12.0, "variance": 3.0 ** 2 - 1}
    tb = GDTSPN.TimedTransition("tb", "B", time_performance=timing_data)
    test_spn.transitions.add(tb)
    timing_data = {"mean": 4.0, "variance": 1.0 ** 2}
    idleb = GDTSPN.TimedTransition("Idle_B", "Idle_B", time_performance=timing_data)
    test_spn.transitions.add(idleb)
    timing_data = {"mean": 8.0, "variance": 3.0 ** 2 - 1}
    tc = GDTSPN.TimedTransition("tc", "C", time_performance=timing_data)
    test_spn.transitions.add(tc)
    timing_data = {"mean": 1.0, "variance": 1.0 ** 2}
    idlec = GDTSPN.TimedTransition("Idle_C", "Idle_C", time_performance=timing_data)
    test_spn.transitions.add(idlec)
    timing_data = {"mean": 9.0, "variance": 2.0 ** 2 - 1}
    td = GDTSPN.TimedTransition("td", "D", time_performance=timing_data)
    test_spn.transitions.add(td)
    timing_data = {"mean": 1.0, "variance": 1.0 ** 2}
    idled = GDTSPN.TimedTransition("Idle_D", "Idle_D", time_performance=timing_data)
    test_spn.transitions.add(idled)
    timing_data = {"mean": 10.0, "variance": 4.0 ** 2 - 1}
    te = GDTSPN.TimedTransition("te", "E", time_performance=timing_data)
    test_spn.transitions.add(te)
    timing_data = {"mean": 5.0, "variance": 1.0 ** 2}
    idlee = GDTSPN.TimedTransition("Idle_E", "Idle_E", time_performance=timing_data)
    test_spn.transitions.add(idlee)
    timing_data = {"mean": 10.0, "variance": 2.0 ** 2 - 1}
    tf = GDTSPN.TimedTransition("tf", "F", time_performance=timing_data)
    test_spn.transitions.add(tf)
    timing_data = {"mean": 1.0, "variance": 1.0 ** 2}
    idlef = GDTSPN.TimedTransition("Idle_F", "Idle_F", time_performance=timing_data)
    test_spn.transitions.add(idlef)
    timing_data = {"mean": 9.0, "variance": 2.0 ** 2 - 1}
    tg = GDTSPN.TimedTransition("tg", "G", time_performance=timing_data)
    test_spn.transitions.add(tg)
    timing_data = {"mean": 1.0, "variance": 1.0 ** 2}
    idleg = GDTSPN.TimedTransition("Idle_G", "Idle_G", time_performance=timing_data)
    test_spn.transitions.add(idleg)
    timing_data = {"mean": 4.0, "variance": 0.71 ** 2}
    th = GDTSPN.TimedTransition("th", "H", time_performance=timing_data)
    test_spn.transitions.add(th)
    timing_data = {"mean": 1, "variance": 0.71 ** 2}
    idleh = GDTSPN.TimedTransition("Idle_H", "Idle_H", time_performance=timing_data)
    test_spn.transitions.add(idleh)

    add_arc_from_to(p_map[0], t1, test_spn)
    add_arc_from_to(t1, p_map[1], test_spn)
    add_arc_from_to(p_map[1], idlea, test_spn)
    add_arc_from_to(idlea, p_map[15], test_spn)
    add_arc_from_to(p_map[15], ta, test_spn)
    add_arc_from_to(ta, p_map[2], test_spn)
    add_arc_from_to(p_map[2], t3, test_spn)
    add_arc_from_to(t3, p_map[3], test_spn)
    add_arc_from_to(t3, p_map[4], test_spn)
    # to C and to D
    add_arc_from_to(p_map[3], idlec, test_spn)
    add_arc_from_to(idlec, p_map[16], test_spn)
    add_arc_from_to(p_map[16], tc, test_spn)

    add_arc_from_to(p_map[4], idled, test_spn)
    add_arc_from_to(idled, p_map[17], test_spn)
    add_arc_from_to(p_map[17], td, test_spn)

    add_arc_from_to(tc, p_map[5], test_spn)
    add_arc_from_to(td, p_map[6], test_spn)

    # lower path
    add_arc_from_to(p_map[0], t2, test_spn)
    add_arc_from_to(t2, p_map[7], test_spn)
    add_arc_from_to(p_map[7], idleb, test_spn)
    add_arc_from_to(idleb, p_map[18], test_spn)
    add_arc_from_to(p_map[18], tb, test_spn)

    add_arc_from_to(tb, p_map[8], test_spn)
    add_arc_from_to(p_map[8], t4, test_spn)
    add_arc_from_to(t4, p_map[9], test_spn)
    add_arc_from_to(t4, p_map[10], test_spn)

    # to E and to F
    add_arc_from_to(p_map[9], idlee, test_spn)
    add_arc_from_to(idlee, p_map[19], test_spn)
    add_arc_from_to(p_map[19], te, test_spn)

    add_arc_from_to(p_map[10], idlef, test_spn)
    add_arc_from_to(idlef, p_map[20], test_spn)
    add_arc_from_to(p_map[20], tf, test_spn)

    add_arc_from_to(te, p_map[5], test_spn)
    add_arc_from_to(tf, p_map[11], test_spn)
    add_arc_from_to(p_map[11], idleg, test_spn)
    add_arc_from_to(idleg, p_map[21], test_spn)
    add_arc_from_to(p_map[21], tg, test_spn)

    add_arc_from_to(tg, p_map[6], test_spn)

    add_arc_from_to(p_map[5], t5, test_spn)
    add_arc_from_to(p_map[6], t5, test_spn)

    # after join
    add_arc_from_to(t5, p_map[12], test_spn)
    add_arc_from_to(p_map[12], t6, test_spn)
    add_arc_from_to(t6, p_map[0], test_spn)

    add_arc_from_to(p_map[12], t7, test_spn)
    add_arc_from_to(t7, p_map[13], test_spn)
    add_arc_from_to(p_map[13], idleh, test_spn)
    add_arc_from_to(idleh, p_map[22], test_spn)
    add_arc_from_to(p_map[22], th, test_spn)

    add_arc_from_to(th, p_map[14], test_spn)

    init_marking = Marking()
    init_marking[p_map[0]] = 1

    final_marking = Marking()
    final_marking[p_map[14]] = 1

    return test_spn, init_marking, final_marking


def create_gdtspn2_no_immediate_variant():
    # Example from Solti (modified)
    test_spn = GDTSPN("Test02 GDT_SPN")
    p_map = create_places(10)
    [test_spn.places.add(p) for p in p_map.values()]

    timing_data = {"mean": 20.0, "variance": 5.0}
    ta = GDTSPN.TimedTransition("ta", "A", time_performance=timing_data)
    test_spn.transitions.add(ta)

    timing_data = {"mean": 16.0, "variance": 3.0}
    tb = GDTSPN.TimedTransition("tb", "B", time_performance=timing_data)
    test_spn.transitions.add(tb)
    timing_data = {"mean": 9.0, "variance": 3.0}
    tc = GDTSPN.TimedTransition("tc", "C", time_performance=timing_data)
    test_spn.transitions.add(tc)
    timing_data = {"mean": 10.0, "variance": 2.0}
    td = GDTSPN.TimedTransition("td", "D", time_performance=timing_data)
    test_spn.transitions.add(td)
    timing_data = {"mean": 15.0, "variance": 4.0}
    te = GDTSPN.TimedTransition("te", "E", time_performance=timing_data)
    test_spn.transitions.add(te)
    timing_data = {"mean": 11.0, "variance": 2.0}
    tf = GDTSPN.TimedTransition("tf", "F", time_performance=timing_data)
    test_spn.transitions.add(tf)
    timing_data = {"mean": 10.0, "variance": 2.0}
    tg = GDTSPN.TimedTransition("tg", "G", time_performance=timing_data)
    test_spn.transitions.add(tg)
    timing_data = {"mean": 5.0, "variance": 1.0}
    th = GDTSPN.TimedTransition("th", "H", time_performance=timing_data)
    test_spn.transitions.add(th)

    tau1 = GDTSPN.Transition("tau1", "")
    test_spn.transitions.add(tau1)
    tau2 = GDTSPN.Transition("tau2", "")
    test_spn.transitions.add(tau2)

    add_arc_from_to(p_map[0], ta, test_spn)
    add_arc_from_to(p_map[0], tb, test_spn)

    add_arc_from_to(ta, p_map[1], test_spn)
    add_arc_from_to(ta, p_map[2], test_spn)

    add_arc_from_to(p_map[1], tc, test_spn)
    add_arc_from_to(p_map[2], td, test_spn)

    add_arc_from_to(tc, p_map[3], test_spn)
    add_arc_from_to(td, p_map[4], test_spn)

    add_arc_from_to(tb, p_map[5], test_spn)
    add_arc_from_to(tb, p_map[6], test_spn)
    add_arc_from_to(p_map[5], te, test_spn)
    add_arc_from_to(te, p_map[3], test_spn)

    add_arc_from_to(p_map[6], tf, test_spn)
    add_arc_from_to(tf, p_map[7], test_spn)
    add_arc_from_to(p_map[7], tg, test_spn)
    add_arc_from_to(tg, p_map[4], test_spn)

    add_arc_from_to(p_map[3], tau1, test_spn)
    add_arc_from_to(p_map[4], tau1, test_spn)
    add_arc_from_to(tau1, p_map[8], test_spn)

    add_arc_from_to(p_map[8], tau2, test_spn)
    add_arc_from_to(tau2, p_map[0], test_spn)
    add_arc_from_to(p_map[8], th, test_spn)
    add_arc_from_to(th, p_map[9], test_spn)

    init_marking = Marking()
    init_marking[p_map[0]] = 1

    final_marking = Marking()
    final_marking[p_map[9]] = 1

    return test_spn, init_marking, final_marking


def create_gdtspn4():
    test_spn = GDTSPN("Test03 GDT_SPN")
    p_map = create_places(7)
    [test_spn.places.add(p) for p in p_map.values()]

    timing_data = {"mean": 20.0, "variance": 5.0 ** 2}
    ta = GDTSPN.TimedTransition("ta", "A", time_performance=timing_data)
    test_spn.transitions.add(ta)
    timing_data = {"mean": 11.0, "variance": 2.0 ** 2}
    tb = GDTSPN.TimedTransition("tb", "B", time_performance=timing_data)
    test_spn.transitions.add(tb)
    timing_data = {"mean": 20.0, "variance": 5.0 ** 2}
    tc = GDTSPN.TimedTransition("tc", "C", time_performance=timing_data)
    test_spn.transitions.add(tc)
    timing_data = {"mean": 11.0, "variance": 2.0 ** 2}
    td = GDTSPN.TimedTransition("td", "D", time_performance=timing_data)
    test_spn.transitions.add(td)
    timing_data = {"mean": 20.0, "variance": 5.0 ** 2}
    te = GDTSPN.TimedTransition("te", "E", time_performance=timing_data)
    test_spn.transitions.add(te)
    timing_data = {"mean": 25.0, "variance": 5.0 ** 2}
    tl = GDTSPN.TimedTransition("tl", "L", time_performance=timing_data)
    test_spn.transitions.add(tl)

    add_arc_from_to(p_map[0], ta, test_spn)
    add_arc_from_to(ta, p_map[1], test_spn)
    add_arc_from_to(ta, p_map[2], test_spn)
    add_arc_from_to(p_map[1], tb, test_spn)
    add_arc_from_to(p_map[2], tc, test_spn)
    add_arc_from_to(tb, p_map[3], test_spn)
    add_arc_from_to(tc, p_map[4], test_spn)
    add_arc_from_to(p_map[3], td, test_spn)
    add_arc_from_to(p_map[4], td, test_spn)
    add_arc_from_to(td, p_map[5], test_spn)
    add_arc_from_to(p_map[5], te, test_spn)
    add_arc_from_to(p_map[5], tl, test_spn)

    add_arc_from_to(te, p_map[6], test_spn)
    add_arc_from_to(tl, p_map[0], test_spn)

    init_marking = Marking()
    init_marking[p_map[0]] = 1

    final_marking = Marking()
    final_marking[p_map[6]] = 1

    return test_spn, init_marking, final_marking


def create_gdtspn5():
    test_spn = GDTSPN("Test03 GDT_SPN")
    p_map = create_places(8)
    [test_spn.places.add(p) for p in p_map.values()]

    timing_data = {"mean": 14.0, "variance": 5.0 ** 2}
    ta = GDTSPN.TimedTransition("ta", "A", time_performance=timing_data)
    test_spn.transitions.add(ta)
    timing_data = {"mean": 8.0, "variance": 2.0 ** 2}
    tb = GDTSPN.TimedTransition("tb", "B", time_performance=timing_data)
    test_spn.transitions.add(tb)
    timing_data = {"mean": 12.0, "variance": 5.0 ** 2}
    tc = GDTSPN.TimedTransition("tc", "C", time_performance=timing_data)
    test_spn.transitions.add(tc)
    timing_data = {"mean": 4.0, "variance": 2.0 ** 2}
    td = GDTSPN.TimedTransition("td", "D", time_performance=timing_data)
    test_spn.transitions.add(td)

    t1 = GDTSPN.ImmediateTransition("t1", "t1", None, None, 1)
    test_spn.transitions.add(t1)
    t2 = GDTSPN.ImmediateTransition("t2", "t2", None, None, 1)
    test_spn.transitions.add(t2)

    add_arc_from_to(p_map[0], ta, test_spn)
    add_arc_from_to(ta, p_map[7], test_spn)
    add_arc_from_to(p_map[7], t1, test_spn)


    add_arc_from_to(t1, p_map[1], test_spn)
    add_arc_from_to(t1, p_map[2], test_spn)

    add_arc_from_to(p_map[1], tb, test_spn)
    add_arc_from_to(p_map[2], tc, test_spn)
    add_arc_from_to(tb, p_map[3], test_spn)
    add_arc_from_to(tc, p_map[4], test_spn)
    add_arc_from_to(p_map[3], t2, test_spn)
    add_arc_from_to(p_map[4], t2, test_spn)
    add_arc_from_to(t2, p_map[5], test_spn)
    add_arc_from_to(p_map[5], td, test_spn)
    add_arc_from_to(td, p_map[6], test_spn)


    init_marking = Marking()
    init_marking[p_map[0]] = 1

    final_marking = Marking()
    final_marking[p_map[6]] = 1

    return test_spn, init_marking, final_marking


def create_gdtspn6():
    # abcd chained net
    test_spn = GDTSPN("Test GDT_SPN")

    p_map = create_places(5)
    [test_spn.places.add(p) for p in p_map.values()]

    timing_data = {"mean": 10.0, "variance": 81.0}
    t1 = GDTSPN.TimedTransition("ta", "A", time_performance=timing_data)
    test_spn.transitions.add(t1)
    timing_data = {"mean": 5.0, "variance": 9.0}
    t2 = GDTSPN.TimedTransition("tb", "B", time_performance=timing_data)
    test_spn.transitions.add(t2)
    timing_data = {"mean": 10.0, "variance": 100.0}
    t3 = GDTSPN.TimedTransition("tc", "C", time_performance=timing_data)
    test_spn.transitions.add(t3)
    timing_data = {"mean": 4.0, "variance": 4.0}
    t4 = GDTSPN.TimedTransition("td", "D", time_performance=timing_data)
    test_spn.transitions.add(t4)

    add_arc_from_to(p_map[0], t1, test_spn)
    add_arc_from_to(t1, p_map[1], test_spn)
    add_arc_from_to(p_map[1], t2, test_spn)
    add_arc_from_to(t2, p_map[2], test_spn)
    add_arc_from_to(p_map[2], t3, test_spn)
    add_arc_from_to(t3, p_map[3], test_spn)
    add_arc_from_to(p_map[3], t4, test_spn)
    add_arc_from_to(t4, p_map[4], test_spn)

    init_marking = Marking()
    init_marking[p_map[0]] = 1

    final_marking = Marking()
    final_marking[p_map[4]] = 1

    return test_spn, init_marking, final_marking


def create_trace1():
    # Create events
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:18", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:27", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:29", "%Y-%m-%d %H:%M:%S")})
    event4 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:01:02", "%Y-%m-%d %H:%M:%S")})
    event5 = Event(
        {"concept:name": "H", "time:timestamp": datetime.strptime("2024-01-01 08:01:07", "%Y-%m-%d %H:%M:%S")})

    # Create a trace and add events
    trace = Trace()
    trace.append(event1)
    trace.append(event2)
    trace.append(event3)
    trace.append(event4)
    trace.append(event5)
    return trace


def create_trace2():
    # Create events
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:22", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:30", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:43", "%Y-%m-%d %H:%M:%S")})
    event4 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:54", "%Y-%m-%d %H:%M:%S")})
    event5 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:01:39", "%Y-%m-%d %H:%M:%S")})
    event6 = Event(
        {"concept:name": "E", "time:timestamp": datetime.strptime("2024-01-01 08:01:59", "%Y-%m-%d %H:%M:%S")})

    # Create a trace and add events
    trace = Trace()
    trace.append(event1)
    trace.append(event2)
    trace.append(event3)
    trace.append(event4)
    trace.append(event5)
    trace.append(event6)
    return trace


def create_trace3():
    # Create events
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:25", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:34", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:35", "%Y-%m-%d %H:%M:%S")})

    # Create a trace and add events
    trace = Trace()
    trace.append(event1)
    trace.append(event2)
    trace.append(event3)
    return trace


def create_trace4():
    # trace for pn_example05
    # Create events
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:11", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:18", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:24", "%Y-%m-%d %H:%M:%S")})
    event4 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:30", "%Y-%m-%d %H:%M:%S")})
    # Create a trace and add events
    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    test_trace.append(event3)
    test_trace.append(event4)
    return test_trace


def create_trace5():  # like example 1 but more realistic timestamps
    # Create events
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:20", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:29", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:30", "%Y-%m-%d %H:%M:%S")})
    event4 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:59", "%Y-%m-%d %H:%M:%S")})
    event5 = Event(
        {"concept:name": "H", "time:timestamp": datetime.strptime("2024-01-01 08:01:05", "%Y-%m-%d %H:%M:%S")})

    # Create a trace and add events
    trace = Trace()
    trace.append(event1)
    trace.append(event2)
    trace.append(event3)
    trace.append(event4)
    trace.append(event5)
    return trace


def create_trace6():  # like example 1 but more realistic timestamps
    # Create events
    event1 = Event(
        {"concept:name": "Idle_A", "time:timestamp": datetime.strptime("2024-01-01 08:00:07", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:20", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "Idle_C", "time:timestamp": datetime.strptime("2024-01-01 08:00:21", "%Y-%m-%d %H:%M:%S")})
    event4 = Event(
        {"concept:name": "Idle_D", "time:timestamp": datetime.strptime("2024-01-01 08:00:22", "%Y-%m-%d %H:%M:%S")})
    event5 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:29", "%Y-%m-%d %H:%M:%S")})
    event6 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:30", "%Y-%m-%d %H:%M:%S")})
    event7 = Event(
        {"concept:name": "Idle_C", "time:timestamp": datetime.strptime("2024-01-01 08:00:52", "%Y-%m-%d %H:%M:%S")})
    event8 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:59", "%Y-%m-%d %H:%M:%S")})
    event9 = Event(
        {"concept:name": "Idle_H", "time:timestamp": datetime.strptime("2024-01-01 08:01:00", "%Y-%m-%d %H:%M:%S")})
    event10 = Event(
        {"concept:name": "H", "time:timestamp": datetime.strptime("2024-01-01 08:01:05", "%Y-%m-%d %H:%M:%S")})

    # Create a trace and add events
    trace = Trace()
    trace.append(event1)
    trace.append(event2)
    trace.append(event3)
    trace.append(event4)
    trace.append(event5)
    trace.append(event6)
    trace.append(event7)
    trace.append(event8)
    trace.append(event9)
    trace.append(event10)
    return trace


def create_trace7():  # like example 5 but more realistic timestamps
    # Create events
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:27", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:35", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:39", "%Y-%m-%d %H:%M:%S")})
    event4 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:01:26", "%Y-%m-%d %H:%M:%S")})
    event5 = Event(
        {"concept:name": "H", "time:timestamp": datetime.strptime("2024-01-01 08:01:34", "%Y-%m-%d %H:%M:%S")})

    # Create a trace and add events
    trace = Trace()
    trace.append(event1)
    trace.append(event2)
    trace.append(event3)
    trace.append(event4)
    trace.append(event5)
    return trace


def create_trace8():
    # trace for pn_example05
    # Create events
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:02", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:12", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:16", "%Y-%m-%d %H:%M:%S")})
    # Create a trace and add events
    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    test_trace.append(event3)
    return test_trace


def create_trace9():
    event1 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:15", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "E", "time:timestamp": datetime.strptime("2024-01-01 08:00:35", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "G", "time:timestamp": datetime.strptime("2024-01-01 08:00:38", "%Y-%m-%d %H:%M:%S")})
    event4 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:01:00", "%Y-%m-%d %H:%M:%S")})
    event5 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:01:09", "%Y-%m-%d %H:%M:%S")})
    event6 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:01:11", "%Y-%m-%d %H:%M:%S")})
    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    test_trace.append(event3)
    test_trace.append(event4)
    test_trace.append(event5)
    test_trace.append(event6)
    return test_trace


def create_trace10():
    event1 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:17", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "E", "time:timestamp": datetime.strptime("2024-01-01 08:00:30", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "F", "time:timestamp": datetime.strptime("2024-01-01 08:00:31", "%Y-%m-%d %H:%M:%S")})
    event4 = Event(
        {"concept:name": "G", "time:timestamp": datetime.strptime("2024-01-01 08:01:15", "%Y-%m-%d %H:%M:%S")})
    event5 = Event(
        {"concept:name": "E", "time:timestamp": datetime.strptime("2024-01-01 08:01:21", "%Y-%m-%d %H:%M:%S")})
    event6 = Event(
        {"concept:name": "H", "time:timestamp": datetime.strptime("2024-01-01 08:01:26", "%Y-%m-%d %H:%M:%S")})

    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    test_trace.append(event3)
    test_trace.append(event4)
    test_trace.append(event5)
    test_trace.append(event6)
    return test_trace


def create_trace11():
    event1 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:17", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "E", "time:timestamp": datetime.strptime("2024-01-01 08:00:30", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:59", "%Y-%m-%d %H:%M:%S")})
    event4 = Event(
        {"concept:name": "E", "time:timestamp": datetime.strptime("2024-01-01 08:01:21", "%Y-%m-%d %H:%M:%S")})
    event5 = Event(
        {"concept:name": "H", "time:timestamp": datetime.strptime("2024-01-01 08:01:26", "%Y-%m-%d %H:%M:%S")})

    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    test_trace.append(event3)
    test_trace.append(event4)
    test_trace.append(event5)
    return test_trace


def create_trace12():
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:04", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:22", "%Y-%m-%d %H:%M:%S")})

    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    return test_trace


def create_trace13():
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:03", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:12", "%Y-%m-%d %H:%M:%S")})

    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    return test_trace


def create_trace14():
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:03", "%Y-%m-%d %H:%M:%S")})

    test_trace = Trace()
    test_trace.append(event1)
    return test_trace


def create_trace15():
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:15", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:28", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:32", "%Y-%m-%d %H:%M:%S")})

    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    test_trace.append(event3)
    return test_trace


def create_trace16():
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:11", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:16", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "C", "time:timestamp": datetime.strptime("2024-01-01 08:00:19", "%Y-%m-%d %H:%M:%S")})

    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    test_trace.append(event3)
    return test_trace


def create_trace17():
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:14", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:23", "%Y-%m-%d %H:%M:%S")})

    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    return test_trace


def create_trace18():
    event1 = Event(
        {"concept:name": "A", "time:timestamp": datetime.strptime("2024-01-01 08:00:14", "%Y-%m-%d %H:%M:%S")})
    event2 = Event(
        {"concept:name": "B", "time:timestamp": datetime.strptime("2024-01-01 08:00:25", "%Y-%m-%d %H:%M:%S")})
    event3 = Event(
        {"concept:name": "D", "time:timestamp": datetime.strptime("2024-01-01 08:00:31", "%Y-%m-%d %H:%M:%S")})

    test_trace = Trace()
    test_trace.append(event1)
    test_trace.append(event2)
    test_trace.append(event3)
    return test_trace


def welcome_message():
    print("Welcome to the Timestamp Reconstructor!")
    print("This tool reconstructs timestamps in a trace for a given GDT_SPN.")
    print("Please use the -m and -t options to specify the model and trace respectively.")


def notify_user():
    time.sleep(15)
    if not process_finished_event.is_set():
        print("The process is still running, please be patient...")


def main():
    parser = argparse.ArgumentParser(description="Please provide the index of the desired GDT_SPN and Trace. \\n"
                                                 "Parameter -m for the GDT_SPN index and parameter -t for the index "
                                                 "of the Trace.")
    parser.add_argument("-m", type=int, choices=range(1, 7), required=True,
                        help="Input for -m must be an integer between 1 and 6")
    parser.add_argument("-t", type=int, choices=range(1, 9), required=True,
                        help="Input for -t must be an integer between 1 and 8")

    # Make sure we are parsing known arguments, sys.argv[1:] contains the arguments from the command line
    args, unknown = parser.parse_known_args(sys.argv[1:])

    # Check if there are unknown arguments or if either -m or -t is missing
    if unknown or not args.m or not args.t:
        print("Both model (-m) and trace (-t) must be specified. See help below.")
        parser.print_help()
        sys.exit(1)

    # Mapping model functions
    model_dict = {
        1: create_gdtspn1,
        2: create_gdtspn2,
        3: create_gdtspn3,
        4: create_gdtspn4,
        5: create_gdtspn5,
        6: create_gdtspn6
    }

    # Mapping trace functions
    trace_dict = {
        1: create_trace1,
        2: create_trace2,
        3: create_trace3,
        4: create_trace4,
        5: create_trace5,
        6: create_trace6,
        7: create_trace7,
        8: create_trace8
    }

    print(f"Selected Model: {args.m}, Selected Trace: {args.t}")
    create_model_function = model_dict.get(args.m)
    if create_model_function:
        gdt_spn, init_marking, final_marking = create_model_function()
    else:
        print("Invalid model number provided. Use -h for help.")
        sys.exit(1)

    create_trace_function = trace_dict.get(args.t)
    if create_trace_function:
        trace = create_trace_function()
    else:
        print("Invalid trace number provided. Use -h for help.")
        sys.exit(1)

    global process_finished_event
    process_finished_event = threading.Event()

    # Start the notification thread
    threading.Thread(target=notify_user).start()

    print(f"Starting to reconstruct timestamps of trace {args.t} and model {args.m}...")
    trace = log_repair.reconstruct_timestamp_with_spn(trace, gdt_spn, init_marking, final_marking)
    print("Finished reconstructing timestamps!")

    process_finished_event.set()


def evaluation():
    # Open a file where you want to store the output
    with open('evaluation/output_gdtspn5.txt', 'w') as f:
        # Redirect stdout to the file
        sys.stdout = f

        gdt_spn, init_marking, final_marking = create_gdtspn5()

        trace_dict = {
            1: create_trace15,
            2: create_trace16,
            3: create_trace17,
            4: create_trace18
        }
        for i in range(1, 5):
            create_trace_function = trace_dict.get(i)
            trace = create_trace_function()
            trace = log_repair.reconstruct_timestamp_with_spn(trace, gdt_spn, init_marking, final_marking)

        # Reset stdout to default to print to console again
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
