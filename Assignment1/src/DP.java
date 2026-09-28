import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DP {

    public static double[] initialValues(MDP m, double init) {
        double[] v = new double[m.numStates()];
        for (int s = 0; s < v.length; s++) {
            v[s] = m.isTerminal(s) ? 0.0 : init;
        }
        return v;
    }

    public static double[][] uniformPolicy(MDP m) {
        double[][] pi = new double[m.numStates()][m.numActions()];
        for (int s = 0; s < m.numStates(); s++) {
            Arrays.fill(pi[s], 1.0 / m.numActions());
        }
        return pi;
    }

    public static double[][] deterministicPolicy(MDP m, int[] policy) {
        double[][] pi = new double[m.numStates()][m.numActions()];
        for (int s = 0; s < m.numStates(); s++) {
            pi[s][policy[s]] = 1.0;
        }
        return pi;
    }

    public static double[] qValues(MDP m, double[] v, double gamma, int s) {
        double[] q = new double[m.numActions()];
        for (int a = 0; a < m.numActions(); a++) {
            Step t = m.step(s, a);
            q[a] = t.reward + gamma * (t.terminal ? 0.0 : v[t.nextState]);
        }
        return q;
    }

    public static int[] greedy(MDP m, double[] v, double gamma) {
        int[] policy = new int[m.numStates()];
        for (int s = 0; s < m.numStates(); s++) {
            if (m.isTerminal(s)) {
                continue;
            }
            double[] q = qValues(m, v, gamma, s);
            int best = 0;
            for (int a = 1; a < q.length; a++) {
                if (q[a] > q[best]) {
                    best = a;
                }
            }
            policy[s] = best;
        }
        return policy;
    }

    public static Result evaluate(MDP m, double[][] pi, double gamma, double tol, double init) {
        Result result = new Result();
        double[] v = initialValues(m, init);
        result.history.add(v.clone());
        while (true) {
            double[] next = new double[m.numStates()];
            double delta = 0.0;
            for (int s = 0; s < m.numStates(); s++) {
                if (m.isTerminal(s)) {
                    next[s] = 0.0;
                    continue;
                }
                double[] q = qValues(m, v, gamma, s);
                double sum = 0.0;
                for (int a = 0; a < q.length; a++) {
                    sum += pi[s][a] * q[a];
                }
                next[s] = sum;
                delta = Math.max(delta, Math.abs(next[s] - v[s]));
            }
            v = next;
            result.sweeps++;
            result.history.add(v.clone());
            if (delta < tol) {
                break;
            }
        }
        result.values = v;
        result.policy = greedy(m, v, gamma);
        return result;
    }

    public static Result policyIteration(MDP m, double gamma, double tol, double init) {
        Result result = new Result();
        double[][] pi = uniformPolicy(m);
        int[] policy = null;
        double[] v = null;
        while (true) {
            Result ev = evaluate(m, pi, gamma, tol, init);
            v = ev.values;
            result.sweeps += ev.sweeps;
            int[] improved = greedy(m, v, gamma);
            result.improvements++;
            if (policy != null && Arrays.equals(policy, improved)) {
                policy = improved;
                break;
            }
            policy = improved;
            pi = deterministicPolicy(m, policy);
        }
        result.values = v;
        result.policy = policy;
        return result;
    }

    public static Result valueIteration(MDP m, double gamma, double tol, double init) {
        Result result = new Result();
        double[] v = initialValues(m, init);
        result.history.add(v.clone());
        while (true) {
            double[] next = new double[m.numStates()];
            double delta = 0.0;
            for (int s = 0; s < m.numStates(); s++) {
                if (m.isTerminal(s)) {
                    next[s] = 0.0;
                    continue;
                }
                double[] q = qValues(m, v, gamma, s);
                double best = q[0];
                for (int a = 1; a < q.length; a++) {
                    best = Math.max(best, q[a]);
                }
                next[s] = best;
                delta = Math.max(delta, Math.abs(next[s] - v[s]));
            }
            v = next;
            result.sweeps++;
            result.history.add(v.clone());
            if (delta < tol) {
                break;
            }
        }
        result.values = v;
        result.policy = greedy(m, v, gamma);
        return result;
    }

    public static List<Integer> rollout(MDP m, int[] policy, int start, int maxSteps) {
        List<Integer> path = new ArrayList<>();
        path.add(start);
        int s = start;
        for (int i = 0; i < maxSteps && !m.isTerminal(s); i++) {
            Step t = m.step(s, policy[s]);
            s = t.nextState;
            path.add(s);
        }
        return path;
    }
}
