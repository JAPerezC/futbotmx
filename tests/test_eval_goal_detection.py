"""Tests del evaluador de detección de goles."""

from __future__ import annotations

from scripts.eval_goal_detection import (
    GoalDetection,
    GoalGT,
    extract_goal_detections,
    match_greedy,
    metrics,
)


def test_extract_fusiona_consecutivos():
    events = [
        {"t": 10.0, "type": "kick"},
        {
            "t": 28.35,
            "type": "goal",
            "meta": {"goal_color": "yellow", "scoring_team": "A"},
        },
        {
            "t": 28.52,
            "type": "goal",
            "meta": {"goal_color": "yellow", "scoring_team": "A"},
        },
        {
            "t": 28.68,
            "type": "goal",
            "meta": {"goal_color": "yellow", "scoring_team": "A"},
        },
        {
            "t": 55.19,
            "type": "goal",
            "meta": {"goal_color": "yellow", "scoring_team": None},
        },
        {
            "t": 57.69,
            "type": "goal",
            "meta": {"goal_color": "yellow", "scoring_team": None},
        },
    ]
    detections = extract_goal_detections(events, merge_window=3.0)
    assert len(detections) == 2
    assert detections[0].n_raw_detections == 3
    assert detections[0].scoring_team == "A"
    assert abs(detections[0].t_start - 28.35) < 1e-6
    assert abs(detections[0].t_end - 28.68) < 1e-6
    # El cluster 55.19/57.69 está dentro de 3 s → un solo cluster.
    assert detections[1].n_raw_detections == 2


def test_extract_separa_si_distancia_mayor():
    events = [
        {"t": 10.0, "type": "goal"},
        {"t": 20.0, "type": "goal"},
        {"t": 30.0, "type": "goal"},
    ]
    detections = extract_goal_detections(events, merge_window=3.0)
    assert len(detections) == 3


def test_extract_sin_goles():
    events = [{"t": 1.0, "type": "kick"}, {"t": 2.0, "type": "pass"}]
    assert extract_goal_detections(events, merge_window=3.0) == []


def test_match_perfecto():
    gt = [GoalGT(t=10.0, goal_color="yellow"), GoalGT(t=30.0, goal_color="yellow")]
    detections = [
        GoalDetection(
            t_start=10.5,
            t_end=10.5,
            goal_color="yellow",
            scoring_team="A",
            n_raw_detections=1,
        ),
        GoalDetection(
            t_start=29.5,
            t_end=29.5,
            goal_color="yellow",
            scoring_team="B",
            n_raw_detections=1,
        ),
    ]
    matched, fn, fp = match_greedy(gt, detections, tolerance=2.0)
    assert len(matched) == 2
    assert fn == []
    assert fp == []


def test_match_con_falsos_positivos():
    gt = [GoalGT(t=10.0, goal_color="yellow")]
    detections = [
        GoalDetection(
            t_start=10.5,
            t_end=10.5,
            goal_color="yellow",
            scoring_team="A",
            n_raw_detections=1,
        ),
        GoalDetection(
            t_start=50.0,
            t_end=50.0,
            goal_color="yellow",
            scoring_team="B",
            n_raw_detections=1,
        ),
    ]
    matched, fn, fp = match_greedy(gt, detections, tolerance=2.0)
    assert len(matched) == 1
    assert fn == []
    assert len(fp) == 1
    assert fp[0].t_start == 50.0


def test_match_con_falsos_negativos():
    gt = [GoalGT(t=10.0, goal_color="yellow"), GoalGT(t=20.0, goal_color="yellow")]
    detections = [
        GoalDetection(
            t_start=10.5,
            t_end=10.5,
            goal_color="yellow",
            scoring_team="A",
            n_raw_detections=1,
        ),
    ]
    matched, fn, fp = match_greedy(gt, detections, tolerance=2.0)
    assert len(matched) == 1
    assert len(fn) == 1
    assert fn[0].t == 20.0
    assert fp == []


def test_match_greedy_elige_mas_cercano():
    gt = [GoalGT(t=10.0, goal_color="yellow")]
    detections = [
        GoalDetection(
            t_start=11.5,
            t_end=11.5,
            goal_color="yellow",
            scoring_team="A",
            n_raw_detections=1,
        ),
        GoalDetection(
            t_start=10.2,
            t_end=10.2,
            goal_color="yellow",
            scoring_team="A",
            n_raw_detections=1,
        ),
    ]
    matched, fn, fp = match_greedy(gt, detections, tolerance=2.0)
    assert len(matched) == 1
    assert abs(matched[0][1].t_center - 10.2) < 1e-6
    # La detección de 11.5 queda como FP
    assert len(fp) == 1


def test_metrics_calculo_basico():
    p, r, f1 = metrics(tp=3, fp=1, fn=1)
    assert abs(p - 0.75) < 1e-6
    assert abs(r - 0.75) < 1e-6
    assert abs(f1 - 0.75) < 1e-6


def test_metrics_sin_predicciones():
    p, r, f1 = metrics(tp=0, fp=0, fn=2)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0
