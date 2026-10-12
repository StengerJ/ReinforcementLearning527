import argparse
import math
import sys

import numpy as np

from stenger_midterm_agent import (
    HERE, HOLD_STEPS, TS, UPRIGHT_RAD, Discretizer, Episode, epsilon_greedy, wrap_to_pi)
from stenger_midterm_experiment import experiment

STAGES = {"balance": "balance", "swingup": "swingup", "comparison": "swingup_comparison"}
DEFAULT_SEED = {"balance": 1, "swingup": 1, "comparison": 0}
ACTION_TORQUE = {0: -0.02, 1: 0.0, 2: 0.02}

W, H = 1100, 680
BG = (18, 22, 30)
GRID = (40, 46, 58)
TEXT = (225, 230, 240)
DIM = (140, 148, 165)
ARM = (90, 160, 255)
ROD = (255, 170, 60)
BOB = (255, 120, 60)
GOOD = (80, 210, 120)
BAD = (235, 80, 80)
WARN = (240, 200, 70)


class Runner:
    def __init__(self, stage, method, seed, eval_seed):
        self.name = STAGES[stage]
        self.task, _ = experiment(self.name)
        path = HERE / "results" / f"{self.name}_{method}_seed{seed}.npz"
        if not path.exists():
            sys.exit(f"missing {path}; run stenger_midterm_experiment.py first")
        q = np.load(path)["q"].copy()
        q.setflags(write=False)
        self.q, self.method, self.seed = q, method, seed
        self.disc = Discretizer(self.task.bins)
        self.reset(eval_seed)

    def reset(self, eval_seed):
        self.eval_seed = eval_seed
        self.rng = np.random.default_rng(eval_seed)
        self.env = self.task.make_env(eval_seed)
        self.ep = Episode(self.task, self.env, self.rng, record=True)
        self.action, self.remaining, self.decisions = 1, 0, 0
        self.done = False

    def tick(self):
        if self.done:
            return
        if self.remaining == 0:
            self.action = epsilon_greedy(self.q[self.disc.index(self.ep.state)], 0.0, self.rng)
            self.remaining = self.task.hold_steps(self.ep.state)
            self.decisions += 1
        _, _, _, self.done = self.ep.sim_step(self.action)
        self.remaining -= 1


def project(p, cx, cy, scale, yaw=math.radians(35), pitch=math.radians(22)):
    x, y, z = p
    sx = x * math.cos(yaw) - y * math.sin(yaw)
    depth = x * math.sin(yaw) + y * math.cos(yaw)
    sy = z * math.cos(pitch) - depth * math.sin(pitch)
    return int(cx + scale * sx), int(cy - scale * sy)


def draw_pendulum(screen, state, cx, cy, scale):
    import pygame
    phi, theta = state[0], state[2]
    l_arm, l_rod = 1.0, 1.15
    base_top = (0.0, 0.0, 0.0)
    ring = [project((l_arm * math.cos(a), l_arm * math.sin(a), 0.0), cx, cy, scale)
            for a in np.linspace(0, 2 * math.pi, 73)]
    pygame.draw.lines(screen, GRID, True, ring, 1)
    for a, col in ((math.pi, BAD), (-math.pi, BAD)):
        pygame.draw.circle(screen, col, project((l_arm * math.cos(a), l_arm * math.sin(a), 0.0), cx, cy, scale), 4)
    pygame.draw.line(screen, (70, 76, 90), project((0, 0, -1.4), cx, cy, scale), project(base_top, cx, cy, scale), 10)
    pygame.draw.ellipse(screen, (60, 66, 80), (*(np.array(project((0, 0, -1.4), cx, cy, scale)) - (60, 14)), 120, 28))
    tip = (l_arm * math.cos(phi), l_arm * math.sin(phi), 0.0)
    t = (-math.sin(phi), math.cos(phi), 0.0)
    bob = (tip[0] + l_rod * math.sin(theta) * t[0], tip[1] + l_rod * math.sin(theta) * t[1],
           l_rod * math.cos(theta))
    up = (tip[0], tip[1], l_rod)
    zone = [project((tip[0] + l_rod * math.sin(a) * t[0], tip[1] + l_rod * math.sin(a) * t[1], l_rod * math.cos(a)),
                    cx, cy, scale) for a in np.linspace(-UPRIGHT_RAD, UPRIGHT_RAD, 15)]
    pygame.draw.lines(screen, (50, 110, 70), False, zone, 3)
    pygame.draw.line(screen, (55, 60, 72), project(tip, cx, cy, scale), project(up, cx, cy, scale), 1)
    pygame.draw.line(screen, ARM, project(base_top, cx, cy, scale), project(tip, cx, cy, scale), 9)
    pygame.draw.circle(screen, ARM, project(tip, cx, cy, scale), 8)
    pygame.draw.line(screen, ROD, project(tip, cx, cy, scale), project(bob, cx, cy, scale), 6)
    pygame.draw.circle(screen, BOB, project(bob, cx, cy, scale), 13)
    pygame.draw.circle(screen, (40, 44, 54), project(base_top, cx, cy, scale), 12)


def draw_trace(screen, font, theta_hist, rect):
    import pygame
    x0, y0, w, h = rect
    pygame.draw.rect(screen, (26, 30, 40), rect)
    for deg in (-180, -90, 0, 90, 180):
        y = y0 + h / 2 - deg / 180 * h / 2
        pygame.draw.line(screen, GRID, (x0, y), (x0 + w, y), 1)
        screen.blit(font.render(f"{deg:+d}", True, DIM), (x0 - 38, y - 8))
    for deg in (-20, 20):
        y = y0 + h / 2 - deg / 180 * h / 2
        pygame.draw.line(screen, (50, 110, 70), (x0, y), (x0 + w, y), 1)
    span = 2000
    pts = []
    for i, th in enumerate(theta_hist[-span:]):
        x = x0 + i / span * w
        y = y0 + h / 2 - math.degrees(th) / 180 * h / 2
        if pts and abs(y - pts[-1][1]) > h / 2:
            if len(pts) > 1:
                pygame.draw.lines(screen, ROD, False, pts, 2)
            pts = []
        pts.append((x, y))
    if len(pts) > 1:
        pygame.draw.lines(screen, ROD, False, pts, 2)
    screen.blit(font.render("wrapped theta (deg) vs time, last 20 s", True, DIM), (x0, y0 - 22))


def draw_hud(screen, font, big, run, paused, speed):
    import pygame
    ep, task = run.ep, run.task
    s = ep.state
    th = math.degrees(wrap_to_pi(s[2]))
    lines = [
        (f"Stage: {run.name}   method: {run.method}", TEXT),
        (f"Q table: train seed {run.seed}   eval seed: {run.eval_seed}   greedy", DIM),
        ("", TEXT),
        (f"time      {ep.log.steps * TS:6.2f} s   (step {ep.log.steps}/{task.max_steps})", TEXT),
        (f"theta~    {th:+7.1f} deg", GOOD if abs(th) <= 20 else TEXT),
        (f"theta_dot {s[3]:+7.2f} rad/s", TEXT),
        (f"phi       {s[0]:+7.2f} rad   (limit +-3.14)", BAD if abs(s[0]) > 2.5 else TEXT),
        (f"phi_dot   {s[1]:+7.2f} rad/s", TEXT),
        (f"action    {run.action}  torque {ACTION_TORQUE[run.action]:+.2f} N m", TEXT),
        (f"control   every {task.hold_steps(s) * 10} ms", DIM),
        (f"return    {ep.log.ret:9.1f}", TEXT),
    ]
    for i, (txt, col) in enumerate(lines):
        screen.blit(font.render(txt, True, col), (640, 24 + i * 24))
    bx, by, bw = 640, 24 + len(lines) * 24 + 10, 400
    frac = min(ep.hold / HOLD_STEPS, 1.0)
    pygame.draw.rect(screen, GRID, (bx, by, bw, 18), 1)
    pygame.draw.rect(screen, GOOD, (bx + 1, by + 1, int((bw - 2) * frac), 16))
    screen.blit(font.render(f"upright hold {ep.hold * TS:4.2f} / 5.00 s   (best {ep.log.max_hold * TS:.2f} s)",
                            True, TEXT), (bx, by + 22))
    if run.done:
        ok = ep.log.max_hold >= HOLD_STEPS
        msg = {"success": "SUCCESS: 5 s upright", "unsafe": "UNSAFE: arm limit |phi| > pi",
               "fell": "FELL: |theta| > 20 deg", "time-limit": "TIME LIMIT: 20 s"}[ep.log.outcome]
        if ok and ep.log.outcome != "success":
            msg = f"SUCCESS: held {ep.log.max_hold * TS:.1f} s ({ep.log.outcome})"
        screen.blit(big.render(msg, True, GOOD if ok else (WARN if ep.log.outcome == "time-limit" else BAD)),
                    (40, 24))
        screen.blit(font.render("R: replay next eval seed    ESC: quit", True, DIM), (40, 62))
    elif paused:
        screen.blit(big.render("PAUSED", True, WARN), (40, 24))
    screen.blit(font.render(f"SPACE pause   +/- speed (x{speed:g})   R restart   ESC quit", True, DIM),
                (40, H - 30))


def run_headless(run):
    while not run.done:
        run.tick()
    report(run)


def report(run):
    log = run.ep.log
    first = "never" if log.first_upright_step < 0 else f"{log.first_upright_step * TS:.2f} s"
    print(f"{run.name} / {run.method} (train seed {run.seed}) / eval seed {run.eval_seed}: outcome={log.outcome}, "
          f"duration={log.steps * TS:.2f} s, return={log.ret:.1f}, first upright={first}, "
          f"longest hold={log.max_hold * TS:.2f} s, decisions={run.decisions}")


def run_window(run, speed, screenshot=None):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Furuta pendulum - Stenger midterm")
    font = pygame.font.SysFont("consolas", 17)
    big = pygame.font.SysFont("consolas", 28, bold=True)
    clock = pygame.time.Clock()
    paused, reported, carry = False, False, 0.0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    run.reset(run.eval_seed + 1 if run.done else run.eval_seed)
                    reported = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed * 2, 16)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed / 2, 0.125)
        dt = clock.tick(60) / 1000.0
        if not paused and not run.done:
            carry += dt * speed / TS
            for _ in range(int(carry)):
                run.tick()
                if run.done:
                    break
            carry -= int(carry)
        if run.done and not reported:
            report(run)
            reported = True
        screen.fill(BG)
        draw_pendulum(screen, run.ep.state, 320, 380, 170)
        draw_trace(screen, font, run.ep.log.theta, (690, 470, 380, 170))
        draw_hud(screen, font, big, run, paused, speed)
        pygame.display.flip()
        if screenshot and run.done:
            pygame.image.save(screen, screenshot)
            pygame.quit()
            return


def main():
    parser = argparse.ArgumentParser(description="Run and visualize one greedy episode with a learned Q table.")
    parser.add_argument("--stage", choices=list(STAGES), default="balance")
    parser.add_argument("--method", choices=["sarsa", "qlearning"], default="qlearning")
    parser.add_argument("--seed", type=int, default=None, help="training seed of the saved Q table (0, 1, 2)")
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    parser.add_argument("--no-render", action="store_true", help="run the episode without a window")
    parser.add_argument("--screenshot", default=None, help="save the final frame to this PNG and exit")
    args = parser.parse_args()
    seed = DEFAULT_SEED[args.stage] if args.seed is None else args.seed
    run = Runner(args.stage, args.method, seed, args.eval_seed)
    if args.no_render:
        run_headless(run)
    else:
        run_window(run, args.speed if not args.screenshot else 16.0, args.screenshot)


if __name__ == "__main__":
    main()
