**English** | [繁體中文](model-derivation.zh-TW.md)

# How this model was derived: a dynamic multi-domain NLS model of a ferroelectric capacitor

This article explains, step by step, how [`spice/fecap_nls.va`](../spice/fecap_nls.va) and its Python reference [`src/fe_models/nls_multidomain.py`](../src/fe_models/nls_multidomain.py) came about. It is written for a reader who meets ferroelectric switching models for the first time. We follow the order of the reasoning: first, which phenomena the model must describe; then a comparison of existing frameworks; then each equation with its physical reason; then where the parameters come from, how the model was verified, how it compares with published data, and finally what it cannot do.

The complete list of equations and the parameter table are in [spice/README.md](../spice/README.md); this article concentrates on *why* each piece is written the way it is.

## Contents

1. [What the model has to describe](#1-what-the-model-has-to-describe)
2. [Comparing candidate frameworks](#2-comparing-candidate-frameworks)
3. [The equations, one step at a time](#3-the-equations-one-step-at-a-time)
4. [Where the parameters come from](#4-where-the-parameters-come-from)
5. [How the model was verified](#5-how-the-model-was-verified)
6. [Comparison with published data](#6-comparison-with-published-data)
7. [Known limitations](#7-known-limitations)
8. [References](#8-references)

## 1. What the model has to describe

The polarization $P$ of a ferroelectric has two stable directions. A large enough voltage flips it into the other direction, and it stays there after the voltage is removed. HfO₂-based films (for example HZO, Hf₀.₅Zr₀.₅O₂) are polycrystalline, and their grains differ in how easily they switch. "How much has switched" is therefore a continuous quantity, and it depends on both voltage and time.

A ferroelectric capacitor model for circuit simulation has to describe at least the following:

- **Partial switching by finite pulses.** At a given voltage a longer pulse switches more; at a given width a higher voltage switches more.
- **Accumulation below the coercive voltage.** A single small pulse switches almost nothing, but after many repetitions the easy-to-switch grains slowly flip. This is a real physical effect.
- **No artificial drift under small signals.** Small symmetric excursions (reads, noise) must not move the state toward zero cycle after cycle. Some models update the state from the voltage path alone and lose a little on every cycle; that drift is a defect of the model, not physics.
- **A state-dependent reversible response.** Without switching, the up and down states have different small-signal capacitance at the same voltage; the measured C–V curve is butterfly-shaped.
- **Imprint.** The hysteresis loop is not centred at 0 V.
- **Leakage**, which depends on polarity, voltage and temperature.
- **Back-switching.** After a write, part of the polarization flips back within microseconds to milliseconds.
- **Scaling with temperature, area and thickness.**

The first two items are the ones most easily confused. Accumulation is real; artificial drift is not. A good model must tell them apart: the former is set by the small group of grains that switch easily, the latter must not exist. Much of the design below serves exactly this distinction.

## 2. Comparing candidate frameworks

Several frameworks describe ferroelectric switching in the literature. We introduce them briefly and then compare them against the requirements above.

- **Miller model** (as discussed in [Wong2005]): the saturated loop plus scaling rules gives $P$ directly from the voltage path.
- **KAI and NLS** (Kolmogorov–Avrami–Ishibashi; nucleation-limited switching) [Jo2007, Alessandri2018]: the film consists of many regions that each wait for a nucleation event and then switch. The original form only describes switching from saturation under a constant voltage.
- **Dynamic multi-domain NLS** [AlessandriIEDM2018, Alessandri2019, Deng2020]: NLS extended to arbitrary waveforms; each region remembers how much switching progress it has accumulated.
- **Preisach model** [Wong2005]: each hysteron has two thresholds and flips when the voltage crosses one; intrinsically rate-independent.
- **Thermally activated (Heracles-type) models** [Fehlings2024, Fehlings2025b]: one state variable with thermally activated rates; distribution and depolarization arise from the voltage division with interface and depletion layers.
- **LK / LGD** (Landau–Khalatnikov, Landau–Ginzburg–Devonshire): a free-energy surface with viscous dynamics.

| Requirement | Miller | NLS / KAI | Dynamic multi-domain NLS | Preisach | Thermally activated | LK / LGD |
|---|---|---|---|---|---|---|
| Switching by finite pulses | no | yes (constant voltage only) | yes (verified over about 5 decades of pulse width [Alessandri2018]) | no (needs an added time response) | yes | partly |
| Accumulation under repeated pulses | tied to the path, hence to the count | no | yes (pulse trains verified [Alessandri2019]) | no (nothing moves without crossing a threshold) | yes (simulated) | not described |
| No drift under small symmetric signals | no (minor loops drift [Wong2005]) | n/a | yes (see §3.6) | yes (by construction) | yes | not described |
| State-dependent reversible response | no | must be added | must be added (per-domain two branches as in [Clima2022]) | must be added | built in, one state for the whole capacitor | built in |
| Leakage | added | added | added | added | built in | partly |
| Back-switching | not described | not described | can be added (§3.5) | not described | from the voltage division | natural only for volatile devices |
| Parameters from common measurements | yes, but cannot express the requirements | yes | yes | yes | partly (interface parameters need TCAD or extra measurements) | difficult |
| SPICE form | simple | needs an integral | original is Monte Carlo, must be rewritten | thresholds must be managed | Verilog-A exists | simple |

**We chose dynamic multi-domain NLS, written as a deterministic mean-field model.** There are three reasons. First, it is the only candidate validated on HZO with constant-voltage switching, minor loops and pulse trains together [AlessandriIEDM2018, Alessandri2019]. Second, its state variables correspond directly to measured quantities (switched fractions). Third, the original Monte Carlo becomes indistinguishable from the analytic NLS at about 5000 grains [Alessandri2019], so for large capacitors an equation for the expectation value can replace the random simulation — the form a circuit simulator needs.

The reversible response, the leakage and the back-switching are added on top of this framework. Thermally activated models are the closest alternative; we did not choose one because its distribution is set by interface parameters that are hard to measure directly. The mechanism of the reversible response is still debated in the literature, so the model only describes the terminal $C(V, \text{state})$ and **does not claim any particular mechanism**.

## 3. The equations, one step at a time

### 3.1 Split the film into $N_d$ classes of domains

As noted above, grains differ in how easily they switch. We sort them into $N_d$ classes by how hard they are to switch. Class $k$ is a fraction $w_k$ of the film ($\sum_k w_k = 1$) and has a state $s_k \in [-1, +1]$: $+1$ means the whole class points up, $-1$ down, $0$ half and half.

The polarization of the capacitor is the weighted mean of the states times the saturation polarization:

$$\bar x = \sum_k w_k s_k, \qquad P = P_s\,\bar x .$$

$\bar x$ is the mean state and $P_s$ the saturation polarization.

### 3.2 The voltage each class sees

Next we need the drive acting on each class. In the simplest case $u_k = V - V_{imp}$: the applied voltage minus a built-in voltage $V_{imp}$, which moves the loop centre away from 0 V (imprint).

Three corrections are added, all off by default:

$$u_k = \rho V - V_{dep}(\bar x - \sigma) - V_{imp} - \delta_k .$$

- $\rho$ is the voltage division by a series interface layer; $\rho = 1$ without one.
- $V_{dep}(\bar x - \sigma)$ is a depolarization voltage, and $\sigma$ a screening state (§3.5).
- $\delta_k$ is a per-class offset (the $E_{off,i}$ of [Deng2020]).

### 3.3 Switching rates: Merz-type and one-directional

A Merz-type law says that the switching time changes exponentially with the field [Alessandri2018]: a stronger field switches faster, and dramatically so. We write it as a rate. Because the voltage only pushes grains toward its own direction, there are two one-directional rates:

$$R_k^{+} = \frac{1}{\tau_0}\exp\left[-\left(\frac{V_{a,k}}{u_k}\right)^{\alpha}\right]\quad (u_k > 0), \qquad R_k^{-} = \frac{1}{\tau_0}\exp\left[-\left(\frac{V_{a,k}}{|u_k|}\right)^{\alpha}\right]\quad (u_k < 0),$$

and the rate in the other direction is zero. $V_{a,k} = E_k t_{fe}$ is the activation voltage of class $k$ ($E_k$ its activation field, $t_{fe}$ the thickness), $\tau_0$ is the attempt time and $\alpha$ an exponent.

This function has an important property: as $u \to 0$ the rate and all its derivatives go smoothly to zero, and the exponent is always negative, so it cannot overflow. Numerically it is enough to set the rate to zero when the exponent is below $-60$.

### 3.4 Weibull switching and the progress variable $h$

Measurements show Weibull switching probabilities $1 - \exp[-(t/\tau)^\beta]$ with $\beta \approx 2$ in HfO₂-based films [Jo2007, Alessandri2018, Alessandri2019]. $\beta > 1$ means that nucleation has an incubation period: switching is slow right after the voltage is applied and then speeds up.

This raises a problem. If the rate depended only on the present voltage, the model would have no memory of how long it has already waited. Each class therefore also keeps an accumulated progress $h_k$ [Alessandri2019, Deng2020]. We use a signed $h_k$ — positive for progress toward +, negative toward − — and write $h_k^{+} = \max(h_k, 0)$, $h_k^{-} = \max(-h_k, 0)$. The state equation is

$$\frac{\mathrm{d}s_k}{\mathrm{d}t} = r_k^{+}(1 - s_k) - r_k^{-}(1 + s_k), \qquad r_k^{\pm} = \beta R_k^{\pm}\,(h_k^{\pm})^{\beta - 1}.$$

The factors $(1 \mp s_k)$ make $s_k = \pm 1$ stable fixed points: a class that has fully switched cannot "over-switch".

**Constant voltage recovers the single-pulse NLS form.** To check that this is right, take the simplest case: start from $s_k = -1$, $h_k = 0$ and apply a constant voltage. Then $h_k = R_k t$, and the equation gives

$$\frac{\mathrm{d}}{\mathrm{d}t}\ln(1 - s_k) = -\beta R_k^{\beta} t^{\beta - 1} \quad\Longrightarrow\quad \frac{1 + s_k}{2} = 1 - \exp\left[-(R_k t)^{\beta}\right].$$

The switched fraction is $f(t) = \sum_k w_k\{1 - \exp[-(R_k t)^\beta]\}$, exactly the single-pulse NLS form of [Alessandri2018]. In other words, under the conditions where the original NLS applies, the dynamic model reduces to it.

**The equation for $h$.** Next we decide how $h$ evolves under a general waveform. It has four parts, each with its own reason:

$$\frac{\mathrm{d}h_k}{\mathrm{d}t} = \underbrace{R_k^{+}\left[1 - \left(\frac{h_k^{+}}{h_m}\right)^{4}\right] - R_k^{-}\left[1 - \left(\frac{h_k^{-}}{h_m}\right)^{4}\right]}_{\text{growth and saturation}} \;\underbrace{-\;\frac{h_k}{\tau_h^{eff}}}_{\text{relaxation without drive}} \;+\; \underbrace{\frac{1}{\tau_x}\left[G(u_k - u_x)\,h_k^{-} - G(-u_k - u_x)\,h_k^{+}\right]}_{\text{reset on reversal}} .$$

- **Growth and saturation.** Under a constant voltage $h$ grows at the rate $R$, so the derivation above holds. $h_m = 5$ only keeps $h$ bounded; the unswitched fraction there is already below $10^{-10}$.
- **Relaxation.** [Alessandri2019] observed that the off time between pulses erases part of the progress and described it by $h \leftarrow \gamma(t_{off})\,h$. We write this as first-order relaxation. The original applies $\gamma$ only during the off time, so the relaxation is gated to voltages close to $V_{imp}$: $\tau_h^{eff} = \tau_h[1 + ((V - V_{imp})/0.1\,\text{V})^4]$. Without the gate, $h$ would also relax during long pulses and the model would no longer reduce to the single-pulse form.
- **Reset on reversal.** When the drive reverses, progress accumulated toward − does not help switching toward +. The reason: grains that are not aligned now were aligned during the previous segment, so they had no progress toward this direction. A fast term ($\tau_x$ = 0.1 ns) therefore clears the opposite $h$. $G(z) = \tfrac12[1 + \tanh(z/10\,\text{mV})]$ is a smooth switch and $u_x = |V_{imp}| + 0.1$ V a threshold, so resting at 0 V ($u = -V_{imp}$) does not count as a reversal. This is an assumption: [Alessandri2019] found that *not* resetting matched periodic stimulation better, while calling that an extreme case.

**Decide "reversal" from the field direction, not from the rate.** This is a bug fixed during development. An earlier version decided whether the drive had reversed from the magnitude of the rate. The consequence: grains in the low-field tail had reverse rates too small to count as reversed, so their $h$ was never reset. In the development records this made the ODE and the closed form differ by 28% for 10⁴ pairs of ±0.67 V reads. Deciding from the field direction brought the difference below 0.5%. The tests in this repository keep the check: with the reset made rate-dependent again, the loss after 300 bipolar ±0.67 V pairs becomes about 2.5 times the closed-form value, and the test fails.

Another development bug was the sign of the reversal term in (E3b): with the wrong sign $h$ diverges exponentially, which the saturated-loop test caught immediately.

**Numerical details.** $h^{\beta - 1}$ is made linear below $h = 10^{-6}$ so that its slope at $h = 0$ stays finite for $\beta < 2$. The reset term is stiff, so the simulator should use Gear integration; in addition the model uses `$bound_step` to keep the time step below 0.2 ns while a reset is in progress (§5).

### 3.5 Back-switching and the screening state $\sigma$

Back-switching after a write comes from the depolarization field pushing domains back [Fehlings2025a, Tian2025]. The simplest form is a constant $V_{dep}\bar x$ in $u_k$. That has a problem: after the write, it keeps pushing forever. The back-switching measured by [Lancaster2022] saturates after about 100 ms, and longer write pulses lose less; the authors attribute this to screening by injected and trapped charge.

We therefore add a screening state, one per capacitor:

$$\tau_s\frac{\mathrm{d}\sigma}{\mathrm{d}t} = \bar x - \sigma ,$$

and write the depolarization term as $V_{dep}(\bar x - \sigma)$. This has three consequences:

- After a long time in one state $\sigma \to \bar x$ and the depolarization term vanishes, so **repeated small-signal reads are not affected**.
- Right after switching $\sigma$ lags behind and back-switching occurs; the longer the write, the more $\sigma$ catches up and the less back-switching.
- Writing to − after a long time in + gives the largest $\bar x - \sigma$ and the strongest push back, which is also the direction of imprint.

$V_{dep}$ is 0 (off) by default. §6 fits $V_{dep}$ and $\tau_s$ to the data of [Lancaster2022].

### 3.6 The activation-field distribution, its sampling, and why small voltages do not accumulate

**Distribution.** [Alessandri2018, Alessandri2019] describe the activation fields by a generalized beta distribution of the second kind (GB2). We use its special case, the Dagum distribution, because its quantile has a closed form that a circuit simulator can evaluate directly:

$$G(E) = \left[1 + (E/b)^{-a}\right]^{-p} \quad\Longleftrightarrow\quad E(q) = b\left[q^{-1/p} - 1\right]^{-1/a}.$$

$b$ is a scale and $a$ sets the width of the core. The low-field tail is $G \approx (E/b)^{ap}$, so $ap$ determines how many grains are particularly easy to switch. This tail is exactly what repeated reads consume. §6 shows that the devices of [Alessandri2018] and [Alessandri2019] have $ap$ of about 87 and 9.9 — very different.

**Sampling.** If the quantile axis were cut into $N_d$ equal parts, the softest part would contain every grain with $q < 1/N_d$ and the tail would not be resolved at all. We therefore place the points on the logit axis $y = \ln[q/(1-q)]$: $y_j = y_m \sinh(a_g\xi_j)/\sinh a_g$ with $\xi_j$ uniform on $[-1, 1]$, $y_m = 9$ and $a_g = 1.5$. The grid is dense in the middle and reaches deep into both tails: with $N_d = 16$ the softest class sits at $q \approx 3\times10^{-4}$ with a weight of about $7\times10^{-4}$. Both directions use the same $q_k$, which assumes that a domain that is hard to switch is hard in both directions.

**Why small voltages do not accumulate.** We can now answer the question from §1. Accumulation is set by the exponent of (E2). For a ±26 mV signal, $u \approx 0.106$ V including the built-in voltage. With $N_d = 16$ the softest class has an activation voltage of 0.78 V, so the exponent is $(0.78/0.106)^{4.11} \approx 3600$ and the rate is numerically zero. Even at ±0.3 V the exponent is about 19 and the rate about $e^{-19}/\tau_0 \approx 0.02\ \text{s}^{-1}$; since $h$ is reset at every reversal, each half cycle contributes only $(Rt)^\beta \sim 10^{-18}$.

This is fundamentally different from a model that updates the state from the voltage path: such a model updates on every cycle and therefore drifts. Here the update is proportional to the time under voltage times an exponentially small rate, and there is no path-driven drift mechanism. The tests keep a counter-example: a state equation that pulls the state toward zero whenever a voltage is applied drifts by more than $10^{-6}$ in five ±0.3 V cycles, and the test fails.

With $N_d = 16$ the part of the tail below the smallest sampled $q$ is merged into the softest class, so the zeros above partly come from the sampling. The model's own bound needs the continuous distribution ($y_m = 40$, $N_d = 4000$). For unipolar 100 ns pulses with 1 µs off, repeated $10^9$ times, the switched fraction is $3\times10^{-10}$ at 30 mV, $4\times10^{-8}$ at 100 mV, $6\times10^{-6}$ at 220 mV and $7\times10^{-5}$ at 300 mV. These bounds extrapolate the shape of the tail (§7).

**Genuine accumulation.** When a read voltage reaches into the tail, the soft domains slowly switch and the loss grows roughly logarithmically with the count. This is not artificial drift: under a unipolar drive it saturates once the soft classes have switched, its size is set by the distribution, and it does not vanish as $N_d \to \infty$ (test: for 10⁴ pairs at ±0.9 V the losses with $N_d$ = 64 and 256 agree within 5%). For small $N_d$ the tail switches class by class, so the loss is a staircase in the count; use $N_d \ge 64$ for slopes.

### 3.7 Reversible charge: four separate terms

The charge that returns when the voltage is removed is split into four terms because their time scales and the ways they are measured differ.

**(1) Linear dielectric:** $q_1 = C_d V$. It does not depend on the state and cancels exactly in a difference between the two states.

**(2) Reversible stretching of the polarization.** Up-polarized domains approach switching when the voltage goes negative: their potential well softens and the susceptibility rises; toward positive voltage they stiffen [Clima2022, Mukherjee2023EDL]. Each domain class therefore follows one of two curves depending on its own state. We use integrated Lorentzians (arctan) for the two branches:

$$g_{\uparrow}(V) = \Delta C\, V_w\left[\arctan\frac{V - V_i + V_s}{V_w} - \arctan\frac{-V_i + V_s}{V_w}\right], \qquad g_{\downarrow}(V) = \Delta C\, V_w\left[\arctan\frac{V - V_i - V_s}{V_w} - \arctan\frac{-V_i - V_s}{V_w}\right].$$

Summed over the classes:

$$q_2 = \sum_k w_k\left[\frac{1 + s_k}{2} g_{\uparrow} + \frac{1 - s_k}{2} g_{\downarrow}\right] = \bar g(V) + \bar x\, g_d(V), \qquad \bar g = \tfrac12(g_\uparrow + g_\downarrow),\ g_d = \tfrac12(g_\uparrow - g_\downarrow).$$

When a domain switches, its reversible contribution moves to the other branch. Because the expression is linear in $s_k$, the sum only needs $\bar x$ and adds no nodes.

This, too, was corrected during development. The first version used one curve $C(V, \bar x)$ for the whole capacitor, with a peak position that moved with $\bar x$. Half-switched, that gives a single high peak at 0 V ($\varepsilon_r \approx 44.8$), whereas the correct per-domain sum is the mean of the two branches ($\approx 34.1$).

$V_i$ is the crossing voltage $V_{CROSS}$ of the two branches. It is a different quantity from the loop centre $V_{imp}$: [Mukherjee2023IEDM] measured the crossing at +0.87 MV/cm while the switching-current peaks are centred near +0.45 MV/cm. A measured C–V "peak" is where a branch, still rising, is cut off by switching; the Lorentzian centre $V_s$ therefore lies beyond the switching point and is never reached by a sweep (see the figure in §6).

**Time scale of (2) and $f_{rev}$.** Term (2) may not be instantaneous. The small-signal window of [Baigol2026] disappears above 2 MHz, but [Mukherjee2023IEDM] still reads a difference with 500 ns pulses. A parameter $f_{rev}$ gives the instantaneous fraction; the rest follows a low-pass filtered voltage $y$:

$$q_2 \;\longrightarrow\; f_{rev}\left[\bar g(V) + \bar x\, g_d(V)\right] + (1 - f_{rev})\left[\bar g(y) + \bar x\, g_d(y)\right], \qquad \tau_r\frac{\mathrm{d}y}{\mathrm{d}t} = V - y .$$

**Why split the whole of (2), not only $\bar x\,g_d$?** If only the state-dependent part is low-pass filtered, the low-pass part has a negative capacitance in one of the states. A square-wave cycle then delivers negative electrical work — the device gives energy back — which violates passivity. Splitting the whole term makes both the instantaneous and the low-pass part positive capacitances, so the device is always passive. The tests check both sides: the correct form gives positive work per cycle for $f_{rev}$ = 0.5 and 0.2; the form that splits only $\bar x\,g_d$ gives negative work, and the test fails. A charge-conservation check cannot catch this error because both forms conserve charge.

**(3) Reversible small motion of domain walls:** $q_3 = C_w(1 - \bar x^2)\,y$. Saturated states have almost no domain walls, hence the factor $(1 - \bar x^2)$; the term does not contribute to the difference between the two saturated states but does for partial states. It follows $y$ because walls are slower than the lattice. A state-independent slow dielectric term $C_r y$ is also kept. Both are off by default.

**(4) Back-switching** is not a separate charge term. It enters through $V_{dep}(\bar x - \sigma)$ of §3.5 and the state equations.

Together:

$$q_{rev} = C_d V + f_{rev}\left[\bar g(V) + \bar x\, g_d(V)\right] + (1 - f_{rev})\left[\bar g(y) + \bar x\, g_d(y)\right] + \left[C_w(1 - \bar x^2) + C_r\right] y .$$

At 0 V with $y = 0$, $q_{rev} = 0$, so the charge left at 0 V is the polarization alone. This lets us read $\bar x$ directly from the charge at 0 V.

### 3.8 Terminal current and leakage

The terminal current is the time derivative of a charge function plus the leakage:

$$I = A\frac{\mathrm{d}}{\mathrm{d}t}\left[P + q_{rev}\right] + A\,J_L(V, T).$$

Writing the charge first and differentiating it has an advantage: charge is conserved automatically.

The leakage consists of mechanisms in parallel, with one parameter set per polarity. The Poole–Frenkel (PF) form follows [Fehlings2024]; Schottky and an effective trap-assisted-tunnelling (TAT) term are also available. One I–V curve at one temperature cannot separate the mechanisms, so only PF and an ohmic term are on by default. The full expression is in [spice/README.md](../spice/README.md#equations).

**A Verilog-A rule: no `ddt` inside a conditional branch.** The model pins the states to their initial values during static analyses and lets them evolve afterwards. An earlier version placed `ddt` inside a "not static" branch; if the simulator takes the history of `ddt` from the DC solution, the initial state is then lost. `ddt` is therefore always written unconditionally, and only the right-hand terms are multiplied by 0 or 1.

## 4. Where the parameters come from

Every default value comes from a published paper or is derived from one; the source of each number is in the [parameter table](../spice/README.md#parameters) and in [`spice/params/generic.inc`](../spice/params/generic.inc). The card is not fitted to any particular device.

- **Switching.** $P_s$, $\alpha$, $\beta$, $\tau_0$ and $V_{imp}$ are from Table I and the text of [Alessandri2019] (W/HZO 8.3 nm/W). The Dagum parameters of the activation field are a fit to switching curves computed from the GB2 distribution published in the same paper (0.8–2.0 V, 200 ns–1 ms). $\tau_h$ is the geometric mean of 1.7 µs and 8.3 µs, the single-exponential times implied by the two relaxation factors of [Alessandri2019] (γ = 0.55 after 1 µs off, 0.3 after 10 µs off); one exponential cannot match both, a known compromise.
- **The choice of $\tau_0$.** An earlier draft combined $\tau_0$ = 2 ns with this distribution. On re-examination, that combination does not reproduce the paper's own switching curves: refitting the distribution with $\tau_0$ fixed at 1–3 ns leaves an RMS error of about 0.09 in switched fraction, against 0.002 with the published 387 ns. Likewise $\beta = 1$ cannot fit (RMS 0.036). The public card therefore uses 387 ns. Devices that switch much faster (for example the 1 ns switching of [Baigol2026]) need their own $\tau_0$ and distribution.
- **Reversible response.** $C_d$, $\Delta C$ and $V_s$ are fitted to C–V values digitized from Fig. 2 (Dev A, symmetric TiN/TiN electrodes) of [Mukherjee2023EDL]; the width $V_w$ is from a fit to Fig. 5a of [Mukherjee2023IEDM] (§6). For symmetric electrodes $V_i \approx 0$.
- **Leakage.** The PF trap depth of 0.68 eV and $\varepsilon$ = 70 are from Table 1 of [Fehlings2024]; the magnitude is scaled to the range digitized from Fig. 6a of [Fehlings2025b] (0.1 A/m² at 2 V).
- **Numbers without a source:** all temperature coefficients (set to 0), the Schottky and TAT defaults (off), the area and thickness (illustrative). They are labelled as such in the parameter table.

## 5. How the model was verified

Verification has three layers. Every number can be reproduced with the commands in the repository.

**Layer 1: unit tests of the Python reference** (`tests/test_nls_multidomain.py`, 33 tests, about 20 s). Highlights:

- Under a constant field the ODE and the single-pulse NLS closed form differ by less than $2\times10^{-4}$; the sampled sum converges to the continuous NLS integral (within $10^{-4}$).
- Closed forms for pulse trains, PUND and bipolar cycles agree with the ODE; a fixed-step integrator converges with the step size.
- Symmetric ±26 mV and ±0.3 V signals do not drift over 10⁶ cycles (closed form) and 20 cycles (ODE); genuine accumulation near the tail converges with $N_d$.
- The butterfly branches, the per-domain sum and the $f_{rev}$ split; passivity for $f_{rev} < 1$.
- Charge conservation in a series-resistor circuit: source charge = change of terminal charge + leakage integral, within $10^{-4}$ of $2P_sA$. The partial derivatives of the charge are written out independently in the test rather than taken from the model, so a missing term in the model makes the check fail.
- **Deliberately broken versions.** The three development bugs mentioned above (low-pass of only $\bar x\,g_d$, a path-driven update, a rate-based reversal) are each implemented as a broken version, and the tests confirm that the corresponding check fails. A check that passes on a broken model is worthless, so this step is necessary.

**Layer 2: a real compiler and simulator.** Before this release the model had only been checked by a hand-written parser and solver. It was now compiled to an OSDI library with OpenVAF-Reloaded 24.0.2 and run in ngspice 47 through 12 testbenches, each compared point by point with the Python reference. The real simulator found one problem:

- The draft pinned the states with `analysis("static")`. The OSDI interface of ngspice does not report `static` during the operating point of a transient analysis. The state equations were then left without a DC condition: ngspice reported "singular matrix", gmin and source stepping failed, and the states started near 0 instead of `s0`. A probe module that only outputs the analysis flags confirmed this (`.op`: static = 1; transient operating point: static = 0, tran = 1). The condition was changed to `analysis("static") || $abstime <= 0`; afterwards every testbench starts from `s0` without warnings.

Everything else (`genvar` loops over electrical arrays, `$bound_step`, unconditional `ddt`) needed no change. The main results (charges in µC/cm², $2P_s$ = 45.8):

| Testbench | Quantity | ngspice vs. Python |
|---|---|---|
| PUND (3 V, 10 µs) | charge at 0 V after each segment | ≤ $1.2\times10^{-5}$ |
| PUND | charge over the whole waveform | ≤ 0.010 |
| PUND | charge conservation inside ngspice (current integral vs. charge change) | $4.1\times10^{-4}$ of $2P_sA$ |
| ±3 V triangles at 1 / 10 / 100 kHz | coercive voltages | ≤ 6.4 mV |
| 24 pulses (1.5–3 V × 0.1–30 µs) | switched fraction | ≤ $1.8\times10^{-4}$ |
| ±0.3 V, 1 MHz, 1000 cycles | change of the mean state | ngspice ≤ $6.5\times10^{-14}$ (round-off), Python 0 |
| ±0.8 V, 1 MHz, 200 cycles (partial state) | genuine slow switching | $-7.22\times10^{-5}$ vs. $-7.12\times10^{-5}$ (1.4%) |
| $f_{rev}$ = 0.5, $C_w$, $C_r$ on | charge over the whole waveform | ≤ 0.0054 |
| back-switching ($V_{dep}$ = 1 V, $\tau_s$ = 100 µs) | back-switched charge within 1 ms | 0.2616 vs. 0.2614 |
| 85 °C, all temperature coefficients non-zero | charge over the whole waveform | ≤ 0.0033 |
| leakage DC sweep (25 and 85 °C) | relative current difference | $7\times10^{-10}$ |

![P-V loops, ngspice vs. Python](spice_pv_loops.png)

![No drift at ±0.3 V; genuine slow switching at ±0.8 V](spice_small_signal.png)

**Layer 3: comparison with published data**, next section.

## 6. Comparison with published data

We found no downloadable raw data of HZO switching kinetics, PUND or P–V, so the comparisons use numbers from the papers: published fit formulas, or points digitized from figures (approximate). They are in [`data/literature/`](../data/literature/); each file starts with the full citation, the figure, and whether the values are a formula or digitized. The figures are produced by [`examples/literature_comparison.py`](../examples/literature_comparison.py) and the numbers stored in `docs/literature_comparison.json`. The figures use the Python reference; the previous section showed that ngspice matches it to within $2\times10^{-4}$ in switched fraction.

**What was fitted, and what was not:** the switching distribution (Dagum parameters) and the two C–V branches are fits; $V_{dep}$ and $\tau_s$ for back-switching are fits; the read-endurance and series-capacitor comparisons fit **no** switching parameter.

**Switching kinetics** [Alessandri2019, Alessandri2018]. Curves computed from the published GB2 fits of both papers (0.8–2.0 V). The Dagum form represents the GB2 of [Alessandri2019] with an RMS error of 0.0022 in switched fraction (continuous distribution) and 0.0065 with $N_d$ = 16; the same refit for [Alessandri2018] gives 0.0003 and 0.0052. So the Dagum form is adequate, and 16 classes are enough. Applying the [Alessandri2019] card unchanged to the [Alessandri2018] device, however, gives an RMS error of 0.26. The two devices differ mainly in the low-field tail ($ap$ about 10 vs. about 87) — precisely the part that sets accumulation under repeated reads.

![Switching kinetics](literature_switching.png)

**The two C–V branches** [Mukherjee2023EDL, Mukherjee2023IEDM]. Three parameters fitted to the digitized points of the symmetric Dev A (width from the IEDM fit) give an RMS of 0.23 in $\varepsilon_r$; these are the card values. Four parameters fitted to the seven points of the asymmetric stack give an RMS of 1.5. The two-branch form reproduces the crossing point and the asymmetric butterfly. It does not reproduce the steep switching-side flank of the measured peaks, because each measured branch ends where that state switches, while the model curve continues.

![C-V](literature_cv.png)

**Read endurance** [Mukherjee2023IEDM Fig. 7]. No switching parameter is fitted; only the activation-field scale is set so that the model's 50 µs P–V loop has the device's coercive voltage, and the tail shape is taken from [Alessandri2019] or [Alessandri2018]. Results:

- At −0.1 V the model is within 0.2 pC (RMS) of the data; this is mainly the C–V fit.
- At −0.3 and −0.5 V the model switches much more than the device during the first reads (−0.5 V, first read: 124 pC with the [Alessandri2019] tail vs. 20.9 pC measured) and then falls too low. The steep [Alessandri2018] tail is closer (RMS 3.5 pC at −0.3 V and 17 pC at −0.5 V for $n \le 10^8$).
- With an illustrative $\tau_0$ = 2 ns the −0.3 V values for $n \ge 10^2$ come within about 2 pC of the data, but the first read gets much worse (−0.5 V: 407 pC).
- After about $10^8$ reads at −0.1 and −0.3 V both states drop together in the data; the model has no such mechanism (fatigue, §7).

Conclusion: read disturb below the coercive voltage is set by the low-field tail and by $\tau_0$, which pulse-switching data above 0.8 V do not determine. Predicting read disturb requires measurements on the device of interest.

![Read endurance](literature_read_endurance.png)

**Back-switching** [Lancaster2022]. 11 nm HZO written at 3.5 V for 25, 100 and 250 µs; digitized values. The fit gives $V_{dep}$ = 1.03 V and $\tau_s$ = 1.7 ms (two parameters, twelve points, RMS 0.19 µC/cm²). With screening the loss saturates and is smaller after longer writes, as in the data; with a constant $V_{dep}$ and no screening the loss keeps growing with delay. The model saturates about a decade earlier than the data and underestimates the loss after 250 µs writes (0.52 vs. 0.75–0.85 µC/cm²).

![Back-switching](literature_backswitching.png)

**Repeated reads through a series capacitor** [Duan2025 Fig. 9]. A qualitative comparison; nothing is fitted. We use the paper's area ratio of 0.83, a 10 nm gate dielectric ($\varepsilon_r$ = 20 assumed) and 2.5 V, 10 µs reads. When the floating node is discharged after every read, the model's state difference shrinks steadily (0.97 → 0.57 after 300 reads); when the node is left floating, it hardly changes (0.97 → 0.966). This is the same ordering as the measured read currents. The mechanism in the model: without discharge, the charge left on the node by the previous switching lowers the voltage across the capacitor in the next read, so switching limits itself.

![Series-capacitor reads](literature_series_read.png)

## 7. Known limitations

| Not described | Consequence |
|---|---|
| Wake-up and fatigue | The model represents one fixed cycling state. [Mukherjee2023IEDM] shows both states dropping together after about $10^8$ reads, which the model cannot produce. $m_P$ and $m_E$ are hooks for a slow state, not implemented. |
| Logarithmic-time imprint | Long storage shifts the coercive voltage by up to hundreds of mV [Tian2025, Buragohain2019]; a single $\tau_s$ cannot describe a logarithmic law. |
| Mechanism of the reversible response | Only the terminal $C(V, \text{state})$ is described; dielectric softening, electrode depletion, interface traps and domain walls are not separated. Extrapolation to other thicknesses, electrodes or frequencies is not guaranteed. |
| Off-time dependence of accumulation | [Alessandri2019] leaves the functional form open; one exponential cannot match both of its points. |
| Stochastic switching of single grains | The model gives expectation values. Small capacitors hold few grains (about 20 in 40 × 40 nm² [AlessandriIEDM2018, Alessandri2019]); the spread of tail switching can exceed the mean signal and needs a Monte Carlo version. |
| Extrapolation | The cited switching data start at 200 ns and 0.8 V. Rates below about 0.8 V and switching faster than 200 ns are extrapolations; measurements there should validate the model rather than be fitted. |
| Polarization-dependent resistance | The sub-300 ps response of [Baigol2026] is not included. |
| Phase changes, antiferroelectric-like pinching | Thin films at elevated temperature can show double-peaked loops [Afroze2026]; the model does not apply then. |
| Temperature coefficients | No quantitative source was found for the linear coefficients, so they default to 0; only the leakage has an Arrhenius dependence. |
| Back-switching after long storage in the opposite state | With $V_{dep}$ on and $\sigma$ starting from $s_0$, the model predicts a large loss within 1 ms after the write, which no data support. $\sigma$ therefore starts at 0 (a cycled film) by default and is meant for short-term back-switching only. |
| Small $\beta$ | The linearization of $h^{\beta-1}$ is negligible for $\beta \gtrsim 1.8$; near $\beta = 1$ it underestimates tail accumulation. |

## 8. References

Only papers whose content was read (full text or the parts indicated) are listed.

- [Jo2007] J. Y. Jo et al., Phys. Rev. Lett. 99, 267602 (2007).
- [Wong2005] Wong, Tsang, Shin, J. Appl. Phys. 98, 074101 (2005), doi:10.1063/1.2060950.
- [Alessandri2018] Alessandri et al., IEEE Electron Device Lett. 39, 1780 (2018), doi:10.1109/LED.2018.2872124.
- [AlessandriIEDM2018] Alessandri, Pandey, Seabaugh, IEDM 2018, doi:10.1109/IEDM.2018.8614607.
- [Alessandri2019] Alessandri et al., IEEE Trans. Electron Devices 66, 3527 (2019), doi:10.1109/TED.2019.2922268.
- [Buragohain2019] Buragohain et al., ACS Appl. Mater. Interfaces 11, 35115 (2019), doi:10.1021/acsami.9b11146.
- [Deng2020] Deng et al., VLSI Technology 2020, doi:10.1109/VLSITechnology18217.2020.9265014.
- [Clima2022] Clima et al., IEEE Trans. Electron Devices 69, 5345 (2022), doi:10.1109/TED.2022.3194822.
- [Lancaster2022] Lancaster et al., Front. Nanotechnol. 4, 939822 (2022), doi:10.3389/fnano.2022.939822.
- [Mukherjee2023EDL] Mukherjee et al., IEEE Electron Device Lett. 44, 1092 (2023), doi:10.1109/LED.2023.3278599.
- [Mukherjee2023IEDM] Mukherjee et al., IEDM 2023, doi:10.1109/IEDM45741.2023.10413879.
- [Fehlings2024] Fehlings et al., arXiv:2410.07791 (v2).
- [Fehlings2025a] Fehlings et al., arXiv:2508.08973.
- [Fehlings2025b] Fehlings et al., arXiv:2511.21267 (v2).
- [Tian2025] Tian, Yang, Liu, arXiv:2512.13748.
- [Duan2025] Duan et al., "Towards Scalable 3D Integration of 2T-nC FeRAM with Hundreds of Layer Stacking", IEDM 2025.
- [Afroze2026] Afroze et al., arXiv:2604.20698.
- [Baigol2026] Baigol et al., arXiv:2606.03677.
