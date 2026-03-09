# QUAD_CORE_vs_SimDiffDiT_vs_HaKAN: Strategy-Spread Analysis Engine 🧠

> **Note:** This project is a direct evolution and continuation of the [QUAD_CORE_AI](https://github.com/OriginalSolutions/QUAD_CORE_AI) repository.

I developed this system as an advanced relational analysis engine. Rather than focusing on simple price prediction, this solution analyzes the **behavior of Balance Curves** across three concurrent strategies: *Main*, *SimDiff*, and *HaKAN*. The project allows me to identify moments when the balance curves of individual strategies diverge, subsequently targeting mean-reversion points.

## 🧠 Research Concept: "Meta-Strategy-Spread Trading"

My meta-strategy does not rely on blind adherence to a single forecast. The system monitors the relationship between three portfolios based on a common financial instrument (BTC), aggregating six distinct models: five specialized AI models and one non-parametric model (Monte Carlo).

1. **Main Strategy:** Integrates *Monte Carlo*, *Random Forest*, *KAN (v2.0)*, and *Neural Trend*. It utilizes adaptive weights, which are continuously trained based on the real-time predictive accuracy of each underlying model.
2. **SimDiff Strategy:** Leverages the *SimDiffDiT* model.
3. **HaKAN Strategy:** Leverages the *HaKAN* model.

---

## 📈 Strategy Logic and Signals

Long-term observations confirm that strategy balance curves exhibit a tendency to diverge and converge. I capitalize on this phenomenon through the following analysis:

1. **Winner vs. Loser Analysis:** I track which strategy is gaining an edge (Winner) and which is currently underperforming (Loser).
2. **Crossover Logic:** I generate a trading signal only when the strategies are in a "counterweight" phase.
3. **Multiplier Algorithm:**
   
    *   **Winning Strategy (Winner):** Assigned a multiplier of **-1**. I treat its signal as a contrarian indicator, anticipating a correction to its over-optimism.
    *   **Losing Strategy (Loser):** Assigned a multiplier of **+1**. I follow its signal, assuming it has reached a pivot point and is primed for a rebound.
    *   **Condition:** A signal is generated **only** when both strategies indicate opposite directions. If they are in agreement, the system remains in a neutral, wait-and-see state.

---

## 🔬 Proprietary AI Model Implementations

I have implemented proprietary solutions that transcend standard machine learning approaches:

### 1. SimDiffDiT (Diffusion Transformer)
Unlike traditional diffusion models, which typically utilize U-Net or convolutional architectures (CNNs), my **SimDiffDiT** utilizes a proprietary **Transformer** layer as its backbone. This architecture significantly improves handling of long-term temporal dependencies in time-series data, which is critical for analyzing financial volatility.
*   **Scientific Inspiration:** Based on advancements in diffusion processes for time-series data: [arXiv:2511.19256v1](https://arxiv.org/html/2511.19256v1).

### 2. HaKAN (Hahn Orthogonal KAN)
My implementation of Kolmogorov-Arnold Networks (KAN) replaces standard weight bases with **Hahn orthogonal polynomials**. Conventional neural networks struggle with discrete market data; my solution allows for extremely precise approximation of the decision function with significantly fewer parameters, resulting in higher model stability when hyperparameters are tuned correctly.
*   **Scientific Inspiration:** Leveraging orthogonal polynomials in the context of KAN: [arXiv:2601.18837v1](https://arxiv.org/html/2601.18837v1).

---

## 🌍 Universal Applications

The architecture I designed is data-agnostic. The "Strategy-Spread" logic proves effective across many domains, depending on whether the system analyzes one or multiple time-series.

### 1. Single-Series Architecture
In this model, the system analyzes relations between the balance curves of AI portfolios powered by a shared data source.

*   **Financial Markets (BTC/USDT):** Leveraging divergence in the adaptation speed of different AI portfolios. The system does not predict price, but reads the spread between strategies to generate a trading signal until the curves converge again.

### 2. Multi-Input Architecture
This model processes two or more independent time-series, learning cross-correlations. This requires adjusting the model's input dimensionality (`input_dim`) to analyze how parameters of one process influence another.

*   **Supply Chain:** Analyzing discrepancies between demand and supply. The system detects when a "Just-in-Time" procurement policy loses efficiency relative to a "Safety Buffer" policy, adjusting inventory levels in real-time.
   
*   **Retail/FMCG:** Detecting moments when an underperforming product group gains ground over another, allowing for dynamic stock optimization.
    
*   **Geothermal Engineering (12km+ boreholes) for supercritical water extraction:** 
    *   **A. Innovative Materials (Filaments):** Production of high-performance, lightweight, and resilient filaments for piping, ensuring maximum supramolecular bonding stability after cooling.
      
    *   **B. Rock Vaporization Methods:** Comparing the efficiency of various drilling techniques (e.g., thermomechanical, chemical, plasma, laser, or microwave). The system monitors "balance curves" (e.g., penetration rate vs. energy cost) of all methods. When one technology approaches its physical limit under specific geological conditions, the system generates a signal to switch to a higher-potential method or optimize process parameters — including **beam configuration, operating frequency, polarization, and power density** — to maintain peak process efficiency.

---

## 🛠 Installation and Execution

1. **Clone the repository:**
   ``` 
    
   git clone https://github.com/OriginalSolutions/QUAD_CORE_vs_SimDiffDiT_vs_HaKAN.git
   cd QUAD_CORE_vs_SimDiffDiT_vs_HaKAN 
   
   ``` 
  
2. **Requirements:** 
   ``` 
    
   pip install -r requirements.txt 
    
   ```

3. **Execution:**
   ``` 
    
   python app.py 
    
   ```

