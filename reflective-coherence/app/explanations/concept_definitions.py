"""
Concept Definitions for Reflective Coherence Explorer

This module provides simplified explanations of key concepts in the
Reflective Coherence (ΨC) framework for users who don't have access
to LLM-generated explanations.

Each concept is explained at three different levels:
- beginner: For users with no background in math or science
- intermediate: For users with some science/math background
- advanced: For users with strong mathematical/technical background
"""

CONCEPT_EXPLANATIONS = {
    "Coherence Accumulation": {
        "beginner": """
        <p>Coherence accumulation is like building a tower of blocks. You add blocks (coherence) over time, 
        but uncertainty (entropy) can shake your tower and slow down your progress.</p>
        
        <p>Imagine saving money in a bank account that earns interest. The more money you have, 
        the more interest you earn (up to a certain point), but unexpected expenses (entropy) 
        reduce how quickly your savings grow.</p>
        
        <p>The system naturally tries to become more coherent over time, but chaos in the environment 
        works against this process.</p>
        """,
        
        "intermediate": """
        <p>Coherence accumulation describes how a system's internal consistency grows over time following a 
        logistic growth pattern, similar to many natural processes like population growth.</p>
        
        <p>The coherence growth is governed by this equation: dC(t)/dt = α*C(t)*(1-C(t)/K) - β*H(M(t))</p>
        
        <p>Where:</p>
        <ul>
            <li>α (alpha) is the coherence growth rate</li>
            <li>K is the maximum possible coherence</li>
            <li>β (beta) measures how strongly entropy affects coherence</li>
            <li>H(M(t)) is the entropy at time t</li>
        </ul>
        
        <p>The growth follows an S-curve that plateaus at K, but entropy continuously 
        slows this growth based on how uncertain or chaotic the environment is.</p>
        """,
        
        "advanced": """
        <p>Coherence accumulation is modeled as a modified logistic growth differential equation with entropy 
        as an inhibiting factor:</p>
        
        <p>dC(t)/dt = α*C(t)*(1-C(t)/K) - β*H(M(t))</p>
        
        <p>This combines the standard logistic growth model (first term) with an entropy-dependent 
        reduction term. The model exhibits several important properties:</p>
        
        <ul>
            <li>When entropy is low, coherence follows standard logistic growth</li>
            <li>When entropy exceeds a critical threshold, coherence growth stalls or reverses</li>
            <li>The equilibrium point is determined by the balance between α, K, β, and the steady-state entropy</li>
            <li>The approach to equilibrium follows sigmoidal dynamics with potential bifurcations at critical entropy values</li>
        </ul>
        
        <p>The phase space of this system reveals attractor states corresponding to coherent and incoherent 
        configurations, with transition dynamics governed by threshold values derived from entropy statistics.</p>
        """
    },
    
    "Entropy Dynamics": {
        "beginner": """
        <p>Entropy is like the messiness or chaos in a system. Think of entropy as how mixed up or unpredictable 
        things are - like a messy room versus a tidy one.</p>
        
        <p>In the ΨC model, entropy works against coherence. The higher the entropy (more chaos or uncertainty), 
        the harder it is for the system to maintain or build coherence.</p>
        
        <p>Imagine trying to build a house of cards in a room where someone occasionally bumps the table. 
        Those bumps are like entropy - they make your task harder and can even undo your progress.</p>
        """,
        
        "intermediate": """
        <p>Entropy dynamics refers to how uncertainty or disorder evolves within the system. Mathematically, 
        we use Shannon entropy to quantify this uncertainty:</p>
        
        <p>H(M(t)) = -∑ p(m_i) * log(p(m_i))</p>
        
        <p>Where p(m_i) is the probability distribution of memory elements or states in the system.</p>
        
        <p>Higher entropy means the system is more disordered or uncertain, making it difficult to maintain 
        coherence. The balance between entropy and coherence determines whether the system can reach and 
        maintain coherent states.</p>
        
        <p>Entropy can change over time in response to external influences or internal reorganization, 
        creating dynamic patterns of coherence growth and decay.</p>
        """,
        
        "advanced": """
        <p>Entropy dynamics in the ΨC framework employs Shannon entropy to quantify uncertainty in the system's state space:</p>
        
        <p>H(M(t)) = -∑<sub>i=1</sub><sup>N</sup> p(m<sub>i</sub>) * log(p(m<sub>i</sub>))</p>
        
        <p>This entropy measure directly impacts coherence accumulation through the coupling parameter β. The relationship 
        between entropy and coherence exhibits several important properties:</p>
        
        <ul>
            <li>Entropy acts as a control parameter that can induce phase transitions in coherence dynamics</li>
            <li>The variance of entropy over time (not just its mean) affects threshold dynamics</li>
            <li>Entropy gradients in the memory state space drive the reflection process that updates the self-model</li>
            <li>The entropy production rate correlates with the system's distance from optimal coherence</li>
        </ul>
        
        <p>For multi-agent systems, the joint entropy across agents creates emergent patterns of collective coherence 
        that cannot be reduced to individual agent dynamics, following principles similar to statistical mechanics 
        of interacting particles.</p>
        """
    },
    
    "Phase Transition Threshold": {
        "beginner": """
        <p>A phase transition threshold is like the temperature at which water freezes or boils - it's the point 
        where a system suddenly changes its behavior.</p>
        
        <p>In the ΨC model, this threshold is the tipping point between a coherent system (organized, stable) 
        and an incoherent one (chaotic, unstable).</p>
        
        <p>Imagine a busy highway - at a certain threshold of traffic density, smooth-flowing traffic suddenly 
        jams up. Similarly, when entropy (chaos) exceeds the threshold, the system's coherence suddenly breaks down.</p>
        """,
        
        "intermediate": """
        <p>The phase transition threshold (θ) marks the boundary between coherent and incoherent states in the system. 
        It's calculated based on the entropy statistics:</p>
        
        <p>θ = E[H(M(t))] + λ<sub>θ</sub> * √Var(H(M(t)))</p>
        
        <p>Where:</p>
        <ul>
            <li>E[H(M(t))] is the expected (average) entropy</li>
            <li>Var(H(M(t))) is the variance of entropy</li>
            <li>λ<sub>θ</sub> is a scaling factor that determines how much variance affects the threshold</li>
        </ul>
        
        <p>When coherence falls below this threshold, the system becomes incoherent. The threshold isn't fixed - 
        it adjusts based on the statistical properties of entropy over time, allowing adaptive responses to 
        changing environments.</p>
        """,
        
        "advanced": """
        <p>The phase transition threshold (θ) represents a critical boundary in the system's state space where 
        coherence dynamics undergo qualitative changes. It's formulated as:</p>
        
        <p>θ = E[H(M(t))] + λ<sub>θ</sub> * √Var(H(M(t)))</p>
        
        <p>This threshold formulation incorporates both first and second-order statistics of entropy, making it 
        sensitive to both the magnitude and variability of uncertainty in the system. The threshold exhibits several 
        important properties:</p>
        
        <ul>
            <li>It self-adjusts based on the entropy distribution, creating an adaptive phase boundary</li>
            <li>The transition across the threshold follows sigmoid dynamics with critical slowing down near the boundary</li>
            <li>The parameter λ<sub>θ</sub> controls the sharpness of the phase transition and the system's sensitivity to entropy fluctuations</li>
            <li>In multi-agent systems, collective thresholds emerge that differ from individual agent thresholds</li>
        </ul>
        
        <p>The mathematical structure of this threshold bears similarity to statistical physics formulations of 
        order parameters in continuous phase transitions, particularly those in non-equilibrium systems with 
        fluctuation-dissipation relationships.</p>
        """
    },
    
    "Reflective Consistency": {
        "beginner": """
        <p>Reflective consistency is like looking in a mirror and making sure your actions match what you believe about yourself. 
        It's about making your self-image line up with your behavior.</p>
        
        <p>In the ΨC model, a system is reflectively consistent when its internal model of itself matches how it 
        actually behaves or functions.</p>
        
        <p>Think of it like a restaurant that claims to provide fast service actually delivering meals quickly. 
        The claim (self-model) and the reality (behavior) are consistent.</p>
        """,
        
        "intermediate": """
        <p>Reflective consistency describes how well a system's internal model of itself aligns with its actual 
        behavior and capabilities. This consistency is what the system optimizes for as it accumulates coherence.</p>
        
        <p>The system maintains reflective consistency by:</p>
        <ul>
            <li>Continuously updating its self-model based on new experiences</li>
            <li>Resolving contradictions between expected and actual outcomes</li>
            <li>Adjusting behavior to align with its self-model, or vice versa</li>
        </ul>
        
        <p>Higher coherence means better reflective consistency, which leads to more accurate predictions and more 
        effective interactions with the environment. When coherence is low, the system struggles to maintain an 
        accurate self-model.</p>
        """,
        
        "advanced": """
        <p>Reflective consistency represents the degree of alignment between a system's self-model and its actual 
        dynamics across state space. It can be formalized through a self-consistency operator that measures the 
        discrepancy between predicted and actual state transitions.</p>
        
        <p>The reflective consistency framework extends traditional notions of self-organization by incorporating 
        higher-order self-reference. Key mathematical properties include:</p>
        
        <ul>
            <li>The self-consistency relation forms a category-theoretic fixed point in the mapping between the 
            system's model space and its behavioral manifold</li>
            <li>The gradient of the self-loss function directly quantifies local deviations from perfect reflective consistency</li>
            <li>The path integral of coherence accumulation over time traces the system's trajectory toward increased 
            reflective consistency</li>
            <li>There exist multiple reflectively consistent states for the same system under different entropy regimes, 
            creating a complex attractor landscape</li>
        </ul>
        
        <p>This formulation provides a rigorous basis for analyzing how cognitive and artificial systems maintain 
        internal consistency while adapting to changing environments, connecting to both information-theoretic and 
        variational principles in complex systems theory.</p>
        """
    },
    
    "Memory Selection": {
        "beginner": """
        <p>Memory selection is like choosing which photos to keep in your album. You pick the ones that tell the 
        most important parts of your story.</p>
        
        <p>In the ΨC model, memory selection is how a system decides which past experiences are most important to 
        remember when updating its understanding of itself.</p>
        
        <p>Imagine a student studying for an exam who focuses on reviewing the most important concepts rather than 
        re-reading the entire textbook. The system similarly prioritizes the most relevant memories.</p>
        """,
        
        "intermediate": """
        <p>Memory selection describes how a system determines which prior states or experiences are most relevant 
        for maintaining coherence. Not all memories are equally important - the system prioritizes those that help 
        resolve contradictions and improve its self-model.</p>
        
        <p>The weight (w<sub>ij</sub>) between two memory states depends on:</p>
        <ul>
            <li>How similar the memories are to each other</li>
            <li>How frequently they're activated together</li>
            <li>How close they are in time</li>
        </ul>
        
        <p>Mathematically: w<sub>ij</sub> = cos(z(m<sub>i</sub>), z(m<sub>j</sub>)) * f<sub>ij</sub> / (1 + α|t<sub>i</sub> - t<sub>j</sub>|)</p>
        
        <p>This memory selection process allows the system to focus computational resources on the most relevant 
        information, improving efficiency and coherence.</p>
        """,
        
        "advanced": """
        <p>Memory selection in the ΨC framework involves optimizing a weighted graph of memory states that 
        maximizes coherence while minimizing computational cost. The weight function between memory states is:</p>
        
        <p>w<sub>ij</sub> = cos(z(m<sub>i</sub>), z(m<sub>j</sub>)) * f<sub>ij</sub> / (1 + α|t<sub>i</sub> - t<sub>j</sub>|)</p>
        
        <p>Where:</p>
        <ul>
            <li>cos(z(m<sub>i</sub>), z(m<sub>j</sub>)) is the cosine similarity between latent representations</li>
            <li>f<sub>ij</sub> is the co-reflection frequency</li>
            <li>α is a temporal decay factor</li>
        </ul>
        
        <p>This formulation connects to several advanced mathematical frameworks:</p>
        
        <ul>
            <li>The memory selection process implements a form of importance sampling in a non-stationary environment</li>
            <li>The resulting memory graph exhibits small-world network properties with high clustering and low path length</li>
            <li>The temporal dynamics follow a power-law forgetting curve modulated by relevance to current coherence needs</li>
            <li>The optimization process solves a constraint satisfaction problem where coherence maximization is balanced 
            against computational efficiency</li>
        </ul>
        
        <p>This memory selection mechanism provides a principled approach to managing the trade-off between complete 
        information and computational tractability, with direct connections to resource-rational models in cognitive science 
        and sparse attention mechanisms in machine learning.</p>
        """
    },
    
    "Self-Loss Function": {
        "beginner": """
        <p>The self-loss function is like a score that tells you how far off your self-understanding is from reality. 
        Lower scores mean your self-image is more accurate.</p>
        
        <p>In the ΨC model, this function measures how much the system's current state differs from its ideal, 
        perfectly coherent state.</p>
        
        <p>Think of it like a GPS that shows how far you are from your destination. The self-loss function shows the 
        system how far it is from optimal coherence, so it knows which direction to go.</p>
        """,
        
        "intermediate": """
        <p>The self-loss function quantifies how much the system's current coherence state deviates from its optimal state. 
        It serves as the objective function that the system tries to minimize through its adaptation process.</p>
        
        <p>Mathematically: L<sub>self</sub> = ∑<sub>i</sub> (C<sub>i</sub>(t) - C<sub>opt</sub>)²</p>
        
        <p>Where:</p>
        <ul>
            <li>C<sub>i</sub>(t) is the coherence of memory m<sub>i</sub> at time t</li>
            <li>C<sub>opt</sub> is the optimal coherence state</li>
        </ul>
        
        <p>The system uses gradient descent on this loss function to adjust its state, gradually moving toward higher 
        coherence. This process is similar to how machine learning models optimize their parameters to minimize error.</p>
        """,
        
        "advanced": """
        <p>The self-loss function provides a geometric interpretation of coherence optimization in the system's state space. 
        It's formulated as:</p>
        
        <p>L<sub>self</sub> = ∑<sub>i</sub> (C<sub>i</sub>(t) - C<sub>opt</sub>)²</p>
        
        <p>This quadratic loss function has several important mathematical properties:</p>
        
        <ul>
            <li>It defines a Riemannian metric on the manifold of possible self-models, with coherence gradients 
            determining the geodesic paths</li>
            <li>The optimization process follows natural gradient descent in this space, accounting for the information 
            geometry of the coherence landscape</li>
            <li>The Hessian of the loss function characterizes the local curvature of the coherence landscape, with 
            eigenvalues indicating sensitivity along different directions</li>
            <li>Local minima in the loss landscape correspond to meta-stable coherent states, while global minima 
            represent optimally coherent configurations</li>
        </ul>
        
        <p>The self-loss function connects to variational free energy principles in cognitive science and machine learning, 
        where systems minimize prediction error while balancing accuracy and complexity. Under certain parametrizations, 
        the self-loss function can be shown to be equivalent to a Kullback-Leibler divergence between the actual and 
        optimal coherence distributions.</p>
        """
    },
    
    "Gradient Descent in Coherence": {
        "beginner": """
        <p>Gradient descent in coherence is like walking downhill to find the lowest point. At each step, you look around 
        and move in the direction that goes down the steepest.</p>
        
        <p>In the ΨC model, the system continuously adjusts itself to become more coherent, always moving in the direction 
        that most improves its internal consistency.</p>
        
        <p>Imagine adjusting the dials on a radio to get the clearest signal. You try different settings and keep the ones 
        that improve the sound, gradually finding the perfect tuning.</p>
        """,
        
        "intermediate": """
        <p>Gradient descent is how the system iteratively optimizes its coherence. It updates its state by moving in the 
        direction that most rapidly decreases the self-loss function.</p>
        
        <p>Mathematically: dC<sub>i</sub>(t)/dt = -∇L<sub>self</sub>(C<sub>i</sub>(t))</p>
        
        <p>This means the system:</p>
        <ul>
            <li>Calculates how changing each part of its state would affect coherence</li>
            <li>Makes adjustments that yield the largest coherence improvements</li>
            <li>Continues this process until reaching optimal coherence or an equilibrium state</li>
        </ul>
        
        <p>The gradient descent process ensures the system efficiently navigates toward higher coherence, resolving 
        contradictions and improving its self-model along the way.</p>
        """,
        
        "advanced": """
        <p>Gradient descent in the coherence framework implements a dynamical system that evolves along the negative 
        gradient of the self-loss function:</p>
        
        <p>dC<sub>i</sub>(t)/dt = -∇L<sub>self</sub>(C<sub>i</sub>(t))</p>
        
        <p>This gradient flow exhibits several mathematically significant properties:</p>
        
        <ul>
            <li>The flow follows the Wasserstein geometry on the space of coherence configurations, minimizing the 
            transport cost between states</li>
            <li>Under appropriate conditions, the convergence rate is O(1/t) for convex regions of the loss landscape 
            and exponential near local minima</li>
            <li>The stochastic version of this gradient flow incorporates noise that scales with entropy, enabling 
            exploration of the coherence landscape proportional to environmental uncertainty</li>
            <li>The system exhibits critical slowing down near phase transitions, with diverging correlation times 
            characteristic of second-order phase transitions</li>
        </ul>
        
        <p>This gradient-based dynamics connects to renormalization group approaches in statistical physics, where 
        the system's behavior at different coherence scales exhibits self-similarity properties. The multi-scale 
        nature of the coherence landscape enables the system to simultaneously optimize local and global coherence 
        through hierarchical gradient descent.</p>
        """
    },
    
    "Multi-Agent Coherence": {
        "beginner": """
        <p>Multi-agent coherence is like a group of musicians playing together in harmony. Each player (agent) adjusts 
        to match the others, creating a coordinated whole that's more than just individual notes.</p>
        
        <p>In the ΨC model, multi-agent coherence describes how multiple systems can align their behaviors and develop 
        shared understanding, creating coherence across the entire group.</p>
        
        <p>Think of a flock of birds that moves together smoothly - no single bird is in charge, but they maintain 
        coherence through simple interactions with their neighbors.</p>
        """,
        
        "intermediate": """
        <p>Multi-agent coherence extends the ΨC framework to systems where coherence emerges from interactions between 
        multiple agents, each with their own internal states and coherence dynamics.</p>
        
        <p>The overall system coherence is a weighted sum of individual agent coherences:</p>
        <p>R(S<sub>t</sub>) = ∑<sub>i=1</sub><sup>N</sup> w<sub>i</sub> · C<sub>i</sub>(t)</p>
        
        <p>Key characteristics include:</p>
        <ul>
            <li>Agents influence each other's coherence through interactions</li>
            <li>Distributed reflection occurs as agents incorporate each other's states into their models</li>
            <li>System-level coherence can emerge even when individual agents have limited information</li>
            <li>Coherence propagates through the network of agent interactions</li>
        </ul>
        
        <p>This framework helps explain how collective intelligence and coordinated behavior emerge in groups without 
        centralized control.</p>
        """,
        
        "advanced": """
        <p>Multi-agent coherence represents a tensorial extension of the single-agent framework, where coherence 
        dynamics unfold in a higher-dimensional space of interacting agents. The system-level coherence is:</p>
        
        <p>R(S<sub>t</sub>) = ∑<sub>i=1</sub><sup>N</sup> w<sub>i</sub> · C<sub>i</sub>(t)</p>
        
        <p>Where w<sub>i</sub> represents the contextual relevance of each agent to the overall system.</p>
        
        <p>The multi-agent framework incorporates several advanced mathematical structures:</p>
        
        <ul>
            <li>The interaction dynamics form a weighted hypergraph where higher-order interactions (beyond pairwise) 
            contribute to emergent coherence</li>
            <li>The system exhibits non-linear phase transitions with critical points that depend on both interaction 
            topology and individual agent parameters</li>
            <li>Information transfer between agents follows a diffusion process governed by the graph Laplacian of 
            the interaction network</li>
            <li>Coherence optimization involves both individual gradient descent and collective alignment forces, 
            creating a multi-scale optimization landscape</li>
        </ul>
        
        <p>This framework connects to statistical field theories where collective modes emerge from local interactions, 
        and to computational theories of consciousness where integrated information arises from distributed processing. 
        Under certain conditions, the multi-agent system demonstrates synchronization phenomena analogous to coupled 
        oscillators, with coherence waves propagating through the agent network.</p>
        """
    }
}

def get_concept_explanation(concept, level="beginner"):
    """
    Get the explanation for a concept at the specified level.
    
    Args:
        concept: The concept to explain
        level: The explanation level (beginner, intermediate, advanced)
        
    Returns:
        HTML-formatted explanation string
    """
    if concept not in CONCEPT_EXPLANATIONS:
        return f"<p>Explanation for '{concept}' not found.</p>"
    
    if level not in CONCEPT_EXPLANATIONS[concept]:
        level = "beginner"  # Default to beginner if level not found
    
    return CONCEPT_EXPLANATIONS[concept][level] 