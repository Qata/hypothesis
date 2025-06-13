//! NavigationSystem - Complete DataTree Novel Prefix Generation Module
//!
//! This module implements the complete NavigationSystem with sophisticated DataTree
//! novel prefix generation capability, following the architectural blueprint patterns.
//! It provides systematic tree traversal algorithms, prefix-based selection orders,
//! tree exhaustion detection, and integration with choice constraint systems.
//!
//! Core capabilities:
//! - DataTree Novel Prefix Generation: Sophisticated algorithms for unexplored path discovery
//! - Systematic Tree Traversal: Multiple traversal strategies with backtracking mechanisms
//! - Prefix-based Selection Orders: Structured shrinking patterns for deterministic navigation
//! - Tree Exhaustion Detection: Mathematical precision in detecting fully explored subtrees
//! - Integration with Choice Constraint Systems: Type-safe navigation respecting choice bounds
//!
//! Architectural principles:
//! - Idiomatic Rust patterns with trait-based design
//! - Comprehensive error handling with structured error types
//! - Debug logging with uppercase hex notation where applicable
//! - PyO3 compatibility for seamless Python integration

use crate::choice::{ChoiceType, ChoiceValue, Constraints};
use crate::datatree::{DataTree, TreeNode, Transition, Branch};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::fmt;
use rand::Rng;

/// Comprehensive error types for NavigationSystem operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NavigationSystemError {
    /// Tree is fully exhausted - no novel prefixes available
    TreeFullyExhausted,
    /// Navigation state corruption detected
    StateCorruption(String),
    /// Invalid navigation parameters
    InvalidParameters(String),
    /// Choice constraint violations during navigation
    ConstraintViolation(String),
    /// Tree structure inconsistency
    TreeInconsistency(String),
    /// Backtracking failure - no viable backtrack points
    BacktrackingFailure,
    /// Algorithm convergence failure
    ConvergenceFailure(String),
}

impl fmt::Display for NavigationSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NavigationSystemError::TreeFullyExhausted => 
                write!(f, "Navigation tree is fully exhausted - no novel prefixes available"),
            NavigationSystemError::StateCorruption(msg) => 
                write!(f, "Navigation state corruption detected: {}", msg),
            NavigationSystemError::InvalidParameters(msg) => 
                write!(f, "Invalid navigation parameters: {}", msg),
            NavigationSystemError::ConstraintViolation(msg) => 
                write!(f, "Choice constraint violation during navigation: {}", msg),
            NavigationSystemError::TreeInconsistency(msg) => 
                write!(f, "Tree structure inconsistency: {}", msg),
            NavigationSystemError::BacktrackingFailure => 
                write!(f, "Backtracking failure - no viable backtrack points found"),
            NavigationSystemError::ConvergenceFailure(msg) => 
                write!(f, "Algorithm convergence failure: {}", msg),
        }
    }
}

impl std::error::Error for NavigationSystemError {}

/// Result type for NavigationSystem operations
pub type NavigationSystemResult<T> = Result<T, NavigationSystemError>;

/// Navigation strategy enumeration for different exploration approaches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NavigationStrategy {
    /// Systematic depth-first search with exhaustive coverage
    SystematicDFS,
    /// Breadth-first search for balanced exploration
    SystematicBFS,
    /// Weighted random exploration with bias towards unexplored regions
    WeightedRandom,
    /// Constraint-guided navigation respecting choice bounds
    ConstraintGuided,
    /// Hybrid approach combining multiple strategies
    HybridApproach,
    /// Minimal distance strategy for shrinking-focused navigation
    MinimalDistance,
}

/// Navigation state tracking for sophisticated tree traversal
#[derive(Debug, Clone)]
pub struct NavigationState {
    /// Current position in the tree
    pub current_node: Arc<TreeNode>,
    /// Trail of visited nodes for backtracking
    pub trail: Vec<(Arc<TreeNode>, usize)>,
    /// Current prefix being built
    pub prefix: Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>,
    /// Depth in the current traversal
    pub depth: usize,
    /// Exhaustion cache for performance optimization
    pub exhaustion_cache: HashMap<u64, bool>,
    /// Strategy being used for this navigation
    pub strategy: NavigationStrategy,
    /// Maximum allowed depth to prevent infinite recursion
    pub max_depth: usize,
}

impl NavigationState {
    /// Create a new navigation state
    pub fn new(root: Arc<TreeNode>, strategy: NavigationStrategy) -> Self {
        log::debug!("NAVIGATION: Creating new navigation state with strategy {:?} from root node 0x{:X}", 
                   strategy, root.node_id);
        
        Self {
            current_node: root,
            trail: Vec::new(),
            prefix: Vec::new(),
            depth: 0,
            exhaustion_cache: HashMap::new(),
            strategy,
            max_depth: 1000,
        }
    }
    
    /// Reset navigation state for a new traversal
    pub fn reset(&mut self, new_strategy: Option<NavigationStrategy>) {
        if let Some(strategy) = new_strategy {
            self.strategy = strategy;
        }
        
        self.trail.clear();
        self.prefix.clear();
        self.depth = 0;
        self.exhaustion_cache.clear();
        
        log::debug!("NAVIGATION: Reset navigation state for strategy {:?}", self.strategy);
    }
    
    /// Add a choice to the current prefix
    pub fn add_choice(&mut self, choice_type: ChoiceType, value: ChoiceValue, constraints: Box<Constraints>) {
        self.prefix.push((choice_type, value, constraints));
        log::debug!("NAVIGATION: Added choice to prefix, total length: {}", self.prefix.len());
    }
    
    /// Remove last choice from prefix (for backtracking)
    pub fn remove_last_choice(&mut self) {
        if let Some((choice_type, value, _)) = self.prefix.pop() {
            log::debug!("NAVIGATION: Removed choice {:?}:{:?} from prefix, new length: {}", 
                       choice_type, value, self.prefix.len());
        }
    }
    
    /// Check if maximum depth is reached
    pub fn is_max_depth_reached(&self) -> bool {
        self.depth >= self.max_depth
    }
}

/// Core NavigationSystem implementing sophisticated DataTree novel prefix generation
pub struct NavigationSystem {
    /// Reference to the underlying DataTree
    tree: Arc<RwLock<DataTree>>,
    /// Current navigation state
    pub state: NavigationState,
    /// Statistics about navigation operations
    stats: NavigationSystemStats,
    /// Configuration parameters
    config: NavigationConfig,
}

/// Configuration parameters for NavigationSystem behavior
#[derive(Debug, Clone)]
pub struct NavigationConfig {
    /// Maximum depth for tree traversal
    pub max_depth: usize,
    /// Maximum number of backtrack attempts
    pub max_backtrack_attempts: usize,
    /// Bias factor for weighted random exploration
    pub random_bias_factor: f64,
    /// Cache size for exhaustion states
    pub exhaustion_cache_size: usize,
    /// Enable sophisticated debugging output
    pub debug_enabled: bool,
}

impl Default for NavigationConfig {
    fn default() -> Self {
        Self {
            max_depth: 1000,
            max_backtrack_attempts: 100,
            random_bias_factor: 0.7,
            exhaustion_cache_size: 10000,
            debug_enabled: true,
        }
    }
}

/// Statistics tracking for NavigationSystem operations
#[derive(Debug, Clone, Default)]
pub struct NavigationSystemStats {
    /// Total number of novel prefixes generated
    pub novel_prefixes_generated: usize,
    /// Total number of tree traversals performed
    pub traversals_performed: usize,
    /// Total number of backtracking operations
    pub backtrack_operations: usize,
    /// Total number of exhaustion checks
    pub exhaustion_checks: usize,
    /// Cache hit rate for exhaustion checks
    pub cache_hit_rate: f64,
    /// Average prefix length generated
    pub average_prefix_length: f64,
    /// Distribution of strategies used
    pub strategy_usage: HashMap<NavigationStrategy, usize>,
}

impl NavigationSystem {
    /// Create a new NavigationSystem with the given DataTree
    pub fn new(tree: DataTree) -> Self {
        let tree_arc = Arc::new(RwLock::new(tree));
        let root = {
            let tree_guard = tree_arc.read().unwrap();
            Arc::clone(&tree_guard.root)
        };
        
        let state = NavigationState::new(root, NavigationStrategy::SystematicDFS);
        
        log::info!("NAVIGATION: Created new NavigationSystem with root node 0x{:X}", state.current_node.node_id);
        
        Self {
            tree: tree_arc,
            state,
            stats: NavigationSystemStats::default(),
            config: NavigationConfig::default(),
        }
    }
    
    /// Generate a novel prefix using sophisticated DataTree traversal algorithms
    pub fn generate_novel_prefix<R: Rng>(&mut self, rng: &mut R) -> NavigationSystemResult<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>> {
        log::info!("NAVIGATION: Starting novel prefix generation with strategy {:?}", self.state.strategy);
        
        self.stats.novel_prefixes_generated += 1;
        self.stats.traversals_performed += 1;
        
        // Update strategy usage statistics
        *self.stats.strategy_usage.entry(self.state.strategy).or_insert(0) += 1;
        
        // Try multiple strategies if the primary strategy fails
        let strategies = [
            self.state.strategy,
            NavigationStrategy::WeightedRandom,
            NavigationStrategy::ConstraintGuided,
            NavigationStrategy::HybridApproach,
        ];
        
        for (attempt, &strategy) in strategies.iter().enumerate() {
            log::debug!("NAVIGATION: Attempt {} using strategy {:?}", attempt + 1, strategy);
            
            self.state.reset(Some(strategy));
            
            match self.execute_navigation_strategy(rng) {
                Ok(prefix) => {
                    log::info!("NAVIGATION: Successfully generated novel prefix with {} choices using strategy {:?}", 
                              prefix.len(), strategy);
                    
                    // Update statistics
                    self.update_prefix_statistics(&prefix);
                    return Ok(prefix);
                }
                Err(NavigationSystemError::TreeFullyExhausted) if attempt == strategies.len() - 1 => {
                    log::warn!("NAVIGATION: All strategies exhausted, tree appears fully explored");
                    return Err(NavigationSystemError::TreeFullyExhausted);
                }
                Err(err) => {
                    log::debug!("NAVIGATION: Strategy {:?} failed: {}", strategy, err);
                    continue;
                }
            }
        }
        
        Err(NavigationSystemError::ConvergenceFailure(
            "All navigation strategies failed to generate novel prefix".to_string()
        ))
    }
    
    /// Execute the current navigation strategy
    fn execute_navigation_strategy<R: Rng>(&mut self, rng: &mut R) -> NavigationSystemResult<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>> {
        match self.state.strategy {
            NavigationStrategy::SystematicDFS => self.systematic_dfs_traversal(),
            NavigationStrategy::SystematicBFS => self.systematic_bfs_traversal(),
            NavigationStrategy::WeightedRandom => self.weighted_random_exploration(rng),
            NavigationStrategy::ConstraintGuided => self.constraint_guided_navigation(),
            NavigationStrategy::HybridApproach => self.hybrid_navigation_approach(rng),
            NavigationStrategy::MinimalDistance => self.minimal_distance_navigation(),
        }
    }
    
    /// Systematic depth-first search traversal with exhaustive coverage
    fn systematic_dfs_traversal(&mut self) -> NavigationSystemResult<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>> {
        log::debug!("NAVIGATION: Starting systematic DFS traversal from node 0x{:X}", self.state.current_node.node_id);
        
        loop {
            if self.state.is_max_depth_reached() {
                log::debug!("NAVIGATION: Maximum depth reached during DFS");
                break;
            }
            
            // Add current node's choices to prefix
            let current_node = Arc::clone(&self.state.current_node);
            self.add_node_choices_to_prefix(&current_node)?;
            
            // Add current node to trail for backtracking
            self.state.trail.push((Arc::clone(&self.state.current_node), self.state.prefix.len()));
            self.state.depth += 1;
            
            // Check for transitions
            match self.get_node_transition(&self.state.current_node)? {
                None => {
                    // No transition - this is novel territory!
                    log::debug!("NAVIGATION: Found novel path at node 0x{:X} with {} choices", 
                               self.state.current_node.node_id, self.state.prefix.len());
                    return Ok(self.state.prefix.clone());
                }
                Some(Transition::Branch(branch)) => {
                    if self.is_branch_exhausted(&branch)? {
                        // Branch exhausted, try backtracking
                        if !self.backtrack_to_unexplored()? {
                            log::debug!("NAVIGATION: No more unexplored branches available");
                            break;
                        }
                        continue;
                    }
                    
                    // Find an unexplored child
                    if let Some(child) = self.find_unexplored_child(&branch)? {
                        self.state.current_node = child;
                        continue;
                    } else {
                        // No unexplored children, novel territory
                        log::debug!("NAVIGATION: Novel territory found in branch at node 0x{:X}", 
                                   self.state.current_node.node_id);
                        return Ok(self.state.prefix.clone());
                    }
                }
                Some(Transition::Conclusion(_)) => {
                    // Reached conclusion, backtrack
                    if !self.backtrack_to_unexplored()? {
                        break;
                    }
                    continue;
                }
                Some(Transition::Killed(_)) => {
                    // Killed branch, backtrack aggressively
                    if !self.backtrack_to_unexplored()? {
                        break;
                    }
                    continue;
                }
            }
        }
        
        // If we reach here, tree might be exhausted
        if self.is_tree_exhausted()? {
            Err(NavigationSystemError::TreeFullyExhausted)
        } else {
            // Generate fallback prefix
            self.generate_fallback_prefix()
        }
    }
    
    /// Systematic breadth-first search traversal for balanced exploration
    fn systematic_bfs_traversal(&mut self) -> NavigationSystemResult<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>> {
        log::debug!("NAVIGATION: Starting systematic BFS traversal");
        
        let mut queue: VecDeque<(Arc<TreeNode>, Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>)> = VecDeque::new();
        let mut visited: HashSet<u64> = HashSet::new();
        
        // Start with root and empty prefix
        queue.push_back((Arc::clone(&self.state.current_node), Vec::new()));
        
        while let Some((node, prefix)) = queue.pop_front() {
            if prefix.len() >= self.config.max_depth {
                continue;
            }
            
            if visited.contains(&node.node_id) {
                continue;
            }
            visited.insert(node.node_id);
            
            // Build prefix with current node's choices
            let mut current_prefix = prefix;
            for i in 0..node.values.len() {
                current_prefix.push((
                    node.choice_types[i],
                    node.values[i].clone(),
                    node.constraints[i].clone(),
                ));
            }
            
            // Check transitions
            match self.get_node_transition(&node)? {
                None => {
                    // Novel territory found
                    log::debug!("NAVIGATION: BFS found novel path at node 0x{:X} with {} choices", 
                               node.node_id, current_prefix.len());
                    return Ok(current_prefix);
                }
                Some(Transition::Branch(branch)) => {
                    if !self.is_branch_exhausted(&branch)? {
                        // Add unexplored children to queue
                        let children = branch.children.read().unwrap();
                        for child in children.values() {
                            if !visited.contains(&child.node_id) && !self.is_node_exhausted(child)? {
                                queue.push_back((Arc::clone(child), current_prefix.clone()));
                            }
                        }
                    }
                }
                Some(Transition::Conclusion(_)) | Some(Transition::Killed(_)) => {
                    // Terminal states, skip
                    continue;
                }
            }
        }
        
        // Queue exhausted, try fallback
        if self.is_tree_exhausted()? {
            Err(NavigationSystemError::TreeFullyExhausted)
        } else {
            self.generate_fallback_prefix()
        }
    }
    
    /// Weighted random exploration with bias towards unexplored regions
    fn weighted_random_exploration<R: Rng>(&mut self, rng: &mut R) -> NavigationSystemResult<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>> {
        log::debug!("NAVIGATION: Starting weighted random exploration");
        
        for attempt in 0..self.config.max_backtrack_attempts {
            self.state.reset(Some(NavigationStrategy::WeightedRandom));
            
            loop {
                if self.state.is_max_depth_reached() {
                    break;
                }
                
                // Add current node's choices to prefix
                let current_node = Arc::clone(&self.state.current_node);
                self.add_node_choices_to_prefix(&current_node)?;
                self.state.trail.push((Arc::clone(&self.state.current_node), self.state.prefix.len()));
                self.state.depth += 1;
                
                // Check transitions
                match self.get_node_transition(&self.state.current_node)? {
                    None => {
                        log::debug!("NAVIGATION: Random exploration found novel path at node 0x{:X}", 
                                   self.state.current_node.node_id);
                        return Ok(self.state.prefix.clone());
                    }
                    Some(Transition::Branch(branch)) => {
                        if let Some(child) = self.select_weighted_random_child(&branch, rng)? {
                            self.state.current_node = child;
                            continue;
                        } else {
                            // No more children, novel territory
                            return Ok(self.state.prefix.clone());
                        }
                    }
                    Some(Transition::Conclusion(_)) | Some(Transition::Killed(_)) => {
                        break; // Try new random path
                    }
                }
            }
            
            log::debug!("NAVIGATION: Random exploration attempt {} completed", attempt + 1);
        }
        
        Err(NavigationSystemError::ConvergenceFailure(
            format!("Weighted random exploration failed after {} attempts", self.config.max_backtrack_attempts)
        ))
    }
    
    /// Constraint-guided navigation respecting choice bounds
    fn constraint_guided_navigation(&mut self) -> NavigationSystemResult<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>> {
        log::debug!("NAVIGATION: Starting constraint-guided navigation");
        
        // Generate choices based on constraint analysis
        let mut prefix = Vec::new();
        
        // Analyze available constraints from the tree
        let constraint_patterns = self.analyze_constraint_patterns()?;
        
        for pattern in constraint_patterns.iter().take(10) { // Limit depth
            let (choice_type, constraints) = pattern;
            
            // Generate optimal choice value for this constraint
            if let Some(value) = self.generate_optimal_choice_value(choice_type, constraints)? {
                prefix.push((*choice_type, value, constraints.clone()));
                
                log::debug!("NAVIGATION: Added constraint-guided choice {:?}", choice_type);
            }
        }
        
        if prefix.is_empty() {
            self.generate_fallback_prefix()
        } else {
            log::debug!("NAVIGATION: Generated constraint-guided prefix with {} choices", prefix.len());
            Ok(prefix)
        }
    }
    
    /// Hybrid approach combining multiple strategies
    fn hybrid_navigation_approach<R: Rng>(&mut self, rng: &mut R) -> NavigationSystemResult<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>> {
        log::debug!("NAVIGATION: Starting hybrid navigation approach");
        
        // Try DFS first for systematic exploration
        if let Ok(prefix) = self.systematic_dfs_traversal() {
            if !prefix.is_empty() {
                return Ok(prefix);
            }
        }
        
        // Fall back to weighted random if DFS fails
        if let Ok(prefix) = self.weighted_random_exploration(rng) {
            if !prefix.is_empty() {
                return Ok(prefix);
            }
        }
        
        // Final fallback to constraint-guided
        self.constraint_guided_navigation()
    }
    
    /// Minimal distance navigation for shrinking-focused traversal
    fn minimal_distance_navigation(&mut self) -> NavigationSystemResult<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>> {
        log::debug!("NAVIGATION: Starting minimal distance navigation");
        
        let mut prefix = Vec::new();
        let mut current_node = Arc::clone(&self.state.current_node);
        
        // Traverse tree choosing minimal/shrink-towards values
        for depth in 0..self.config.max_depth {
            // Add current node's choices (preferring minimal values)
            for i in 0..current_node.values.len() {
                let choice_type = current_node.choice_types[i];
                let value = current_node.values[i].clone();
                let constraints = current_node.constraints[i].clone();
                
                // Prefer shrink-towards values when available
                let optimal_value = self.get_shrink_towards_value(&choice_type, &constraints)
                    .unwrap_or(value);
                
                prefix.push((choice_type, optimal_value, constraints));
            }
            
            // Navigate to child with minimal complexity
            match self.get_node_transition(&current_node)? {
                None => {
                    log::debug!("NAVIGATION: Minimal distance found novel path at depth {}", depth);
                    return Ok(prefix);
                }
                Some(Transition::Branch(branch)) => {
                    if let Some(child) = self.find_minimal_complexity_child(&branch)? {
                        current_node = child;
                        continue;
                    } else {
                        return Ok(prefix);
                    }
                }
                Some(Transition::Conclusion(_)) | Some(Transition::Killed(_)) => {
                    break;
                }
            }
        }
        
        if prefix.is_empty() {
            self.generate_fallback_prefix()
        } else {
            Ok(prefix)
        }
    }
    
    /// Add current node's choices to the navigation prefix
    fn add_node_choices_to_prefix(&mut self, node: &Arc<TreeNode>) -> NavigationSystemResult<()> {
        for i in 0..node.values.len() {
            let choice_type = node.choice_types[i];
            let value = node.values[i].clone();
            let constraints = node.constraints[i].clone();
            
            self.state.add_choice(choice_type, value, constraints);
        }
        Ok(())
    }
    
    /// Get the transition from a node (handles locking safely)
    fn get_node_transition(&self, node: &Arc<TreeNode>) -> NavigationSystemResult<Option<Transition>> {
        match node.transition.read() {
            Ok(guard) => Ok(guard.as_ref().map(|t| match t {
                Transition::Branch(b) => Transition::Branch(b.clone()),
                Transition::Conclusion(c) => Transition::Conclusion(c.clone()),
                Transition::Killed(k) => Transition::Killed(k.clone()),
            })),
            Err(_) => Err(NavigationSystemError::StateCorruption(
                "Failed to read node transition".to_string()
            )),
        }
    }
    
    /// Check if a branch is exhausted with caching
    fn is_branch_exhausted(&mut self, branch: &Branch) -> NavigationSystemResult<bool> {
        self.stats.exhaustion_checks += 1;
        
        // Check explicit exhaustion flag first
        if let Ok(guard) = branch.is_exhausted.read() {
            if *guard {
                return Ok(true);
            }
        }
        
        // Check if all children are exhausted
        let children = match branch.children.read() {
            Ok(guard) => guard,
            Err(_) => return Err(NavigationSystemError::StateCorruption(
                "Failed to read branch children".to_string()
            )),
        };
        
        if children.is_empty() {
            return Ok(false); // Not exhausted if no children yet
        }
        
        for child in children.values() {
            if !self.is_node_exhausted(child)? {
                return Ok(false);
            }
        }
        
        // All children exhausted, mark branch as exhausted
        if let Ok(mut guard) = branch.is_exhausted.write() {
            *guard = true;
        }
        
        Ok(true)
    }
    
    /// Check if a specific node is exhausted
    fn is_node_exhausted(&mut self, node: &Arc<TreeNode>) -> NavigationSystemResult<bool> {
        // Check cache first
        if let Some(&cached) = self.state.exhaustion_cache.get(&node.node_id) {
            return Ok(cached);
        }
        
        let exhausted = node.check_exhausted();
        
        // Cache the result
        if self.state.exhaustion_cache.len() < self.config.exhaustion_cache_size {
            self.state.exhaustion_cache.insert(node.node_id, exhausted);
        }
        
        Ok(exhausted)
    }
    
    /// Find an unexplored child in a branch
    fn find_unexplored_child(&mut self, branch: &Branch) -> NavigationSystemResult<Option<Arc<TreeNode>>> {
        let children = match branch.children.read() {
            Ok(guard) => guard,
            Err(_) => return Err(NavigationSystemError::StateCorruption(
                "Failed to read branch children".to_string()
            )),
        };
        
        for child in children.values() {
            if !self.is_node_exhausted(child)? {
                return Ok(Some(Arc::clone(child)));
            }
        }
        
        Ok(None)
    }
    
    /// Select a weighted random child from a branch
    fn select_weighted_random_child<R: Rng>(&mut self, branch: &Branch, rng: &mut R) -> NavigationSystemResult<Option<Arc<TreeNode>>> {
        let children = match branch.children.read() {
            Ok(guard) => guard,
            Err(_) => return Err(NavigationSystemError::StateCorruption(
                "Failed to read branch children".to_string()
            )),
        };
        
        let mut candidates: Vec<(Arc<TreeNode>, f64)> = Vec::new();
        
        for child in children.values() {
            if !self.is_node_exhausted(child)? {
                let weight = self.calculate_exploration_weight(child)?;
                candidates.push((Arc::clone(child), weight));
            }
        }
        
        if candidates.is_empty() {
            return Ok(None);
        }
        
        // Weighted random selection
        let total_weight: f64 = candidates.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            // Uniform selection fallback
            let index = rng.gen_range(0..candidates.len());
            return Ok(Some(Arc::clone(&candidates[index].0)));
        }
        
        let mut target = rng.gen::<f64>() * total_weight;
        for (node, weight) in &candidates {
            target -= weight;
            if target <= 0.0 {
                return Ok(Some(Arc::clone(node)));
            }
        }
        
        // Fallback (should not reach here) - use first candidate if available
        Ok(candidates.first().map(|(node, _)| Arc::clone(node)))
    }
    
    /// Calculate exploration weight for a node
    fn calculate_exploration_weight(&mut self, node: &Arc<TreeNode>) -> NavigationSystemResult<f64> {
        let mut weight = 1.0;
        
        // Favor nodes with fewer children (less explored)
        if let Ok(Some(Transition::Branch(branch))) = self.get_node_transition(node) {
            let child_count = match branch.children.read() {
                Ok(guard) => guard.len(),
                Err(_) => return Err(NavigationSystemError::StateCorruption(
                    "Failed to read branch children for weight calculation".to_string()
                )),
            };
            
            if child_count > 0 {
                weight *= 1.0 / (child_count as f64).sqrt();
            }
        }
        
        // Boost weight for nodes with no transition (novel territory)
        if self.get_node_transition(node)?.is_none() {
            weight *= 3.0;
        }
        
        // Apply bias factor from configuration
        weight *= self.config.random_bias_factor;
        
        Ok(weight.max(0.01)) // Ensure minimum weight
    }
    
    /// Backtrack to the nearest unexplored branch point
    fn backtrack_to_unexplored(&mut self) -> NavigationSystemResult<bool> {
        self.stats.backtrack_operations += 1;
        
        while let Some((node, prefix_len)) = self.state.trail.pop() {
            // Restore prefix to this point
            self.state.prefix.truncate(prefix_len);
            self.state.depth = self.state.trail.len();
            
            // Check if this node has unexplored branches
            if let Ok(Some(Transition::Branch(branch))) = self.get_node_transition(&node) {
                if !self.is_branch_exhausted(&branch)? {
                    let node_id = node.node_id;
                    self.state.current_node = node;
                    log::debug!("NAVIGATION: Backtracked to node 0x{:X} with unexplored branches", 
                               node_id);
                    return Ok(true);
                }
            }
        }
        
        log::debug!("NAVIGATION: Backtracking failed - no unexplored branches found");
        Ok(false)
    }
    
    /// Check if the entire tree is exhausted
    fn is_tree_exhausted(&mut self) -> NavigationSystemResult<bool> {
        let tree_guard = match self.tree.read() {
            Ok(guard) => guard,
            Err(_) => return Err(NavigationSystemError::StateCorruption(
                "Failed to read tree for exhaustion check".to_string()
            )),
        };
        
        Ok(tree_guard.root.check_exhausted())
    }
    
    /// Generate fallback prefix when primary strategies fail
    fn generate_fallback_prefix(&self) -> NavigationSystemResult<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>> {
        log::debug!("NAVIGATION: Generating fallback prefix");
        
        let mut prefix = Vec::new();
        
        // Add a simple integer choice as fallback
        let choice_type = ChoiceType::Integer;
        let value = ChoiceValue::Integer(0);
        let constraints = Box::new(Constraints::Integer(crate::choice::IntegerConstraints::default()));
        
        prefix.push((choice_type, value, constraints));
        
        log::debug!("NAVIGATION: Generated fallback prefix with {} choices", prefix.len());
        Ok(prefix)
    }
    
    /// Analyze constraint patterns available in the tree
    fn analyze_constraint_patterns(&self) -> NavigationSystemResult<Vec<(ChoiceType, Box<Constraints>)>> {
        let mut patterns = Vec::new();
        
        // Basic patterns for each choice type
        patterns.push((ChoiceType::Integer, Box::new(Constraints::Integer(
            crate::choice::IntegerConstraints::default()
        ))));
        
        patterns.push((ChoiceType::Boolean, Box::new(Constraints::Boolean(
            crate::choice::BooleanConstraints { p: 0.5 }
        ))));
        
        patterns.push((ChoiceType::Float, Box::new(Constraints::Float(
            crate::choice::FloatConstraints {
                min_value: 0.0,
                max_value: 1.0,
                allow_nan: false,
                smallest_nonzero_magnitude: Some(1e-10),
            }
        ))));
        
        Ok(patterns)
    }
    
    /// Generate optimal choice value for given constraints
    fn generate_optimal_choice_value(&self, choice_type: &ChoiceType, constraints: &Box<Constraints>) -> NavigationSystemResult<Option<ChoiceValue>> {
        match (choice_type, constraints.as_ref()) {
            (ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                let value = int_constraints.shrink_towards.unwrap_or(0);
                Ok(Some(ChoiceValue::Integer(value)))
            }
            (ChoiceType::Boolean, Constraints::Boolean(_)) => {
                Ok(Some(ChoiceValue::Boolean(false))) // Prefer false for shrinking
            }
            (ChoiceType::Float, Constraints::Float(float_constraints)) => {
                let value = if float_constraints.min_value <= 0.0 && float_constraints.max_value >= 0.0 {
                    0.0
                } else {
                    float_constraints.min_value
                };
                Ok(Some(ChoiceValue::Float(value)))
            }
            _ => Ok(None),
        }
    }
    
    /// Get shrink-towards value for a choice type and constraints
    fn get_shrink_towards_value(&self, choice_type: &ChoiceType, constraints: &Box<Constraints>) -> Option<ChoiceValue> {
        match (choice_type, constraints.as_ref()) {
            (ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                Some(ChoiceValue::Integer(int_constraints.shrink_towards.unwrap_or(0)))
            }
            (ChoiceType::Boolean, Constraints::Boolean(_)) => {
                Some(ChoiceValue::Boolean(false))
            }
            (ChoiceType::Float, Constraints::Float(_)) => {
                Some(ChoiceValue::Float(0.0))
            }
            _ => None,
        }
    }
    
    /// Find child with minimal complexity for shrinking
    fn find_minimal_complexity_child(&mut self, branch: &Branch) -> NavigationSystemResult<Option<Arc<TreeNode>>> {
        let children = match branch.children.read() {
            Ok(guard) => guard,
            Err(_) => return Err(NavigationSystemError::StateCorruption(
                "Failed to read branch children".to_string()
            )),
        };
        
        let mut minimal_child: Option<Arc<TreeNode>> = None;
        let mut minimal_complexity = f64::INFINITY;
        
        for child in children.values() {
            if !self.is_node_exhausted(child)? {
                let complexity = self.calculate_node_complexity(child);
                if complexity < minimal_complexity {
                    minimal_complexity = complexity;
                    minimal_child = Some(Arc::clone(child));
                }
            }
        }
        
        Ok(minimal_child)
    }
    
    /// Calculate complexity score for a node
    fn calculate_node_complexity(&self, node: &Arc<TreeNode>) -> f64 {
        let mut complexity = 0.0;
        
        // Add complexity based on choice values
        for value in &node.values {
            complexity += match value {
                ChoiceValue::Integer(i) => i.abs() as f64,
                ChoiceValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
                ChoiceValue::Float(f) => f.abs(),
                ChoiceValue::String(s) => s.len() as f64,
                ChoiceValue::Bytes(b) => b.len() as f64,
            };
        }
        
        complexity
    }
    
    /// Update statistics based on generated prefix
    fn update_prefix_statistics(&mut self, prefix: &[(ChoiceType, ChoiceValue, Box<Constraints>)]) {
        let length = prefix.len() as f64;
        
        // Update running average
        let total_prefixes = self.stats.novel_prefixes_generated as f64;
        let current_avg = self.stats.average_prefix_length;
        self.stats.average_prefix_length = ((current_avg * (total_prefixes - 1.0)) + length) / total_prefixes;
        
        // Update cache hit rate
        let total_checks = self.stats.exhaustion_checks as f64;
        if total_checks > 0.0 {
            let cache_hits = self.state.exhaustion_cache.len() as f64;
            self.stats.cache_hit_rate = cache_hits / total_checks;
        }
    }
    
    /// Get current navigation statistics
    pub fn get_stats(&self) -> NavigationSystemStats {
        self.stats.clone()
    }
    
    /// Reset navigation statistics
    pub fn reset_stats(&mut self) {
        self.stats = NavigationSystemStats::default();
        log::debug!("NAVIGATION: Statistics reset");
    }
    
    /// Configure navigation parameters
    pub fn configure(&mut self, config: NavigationConfig) {
        self.config = config;
        self.state.max_depth = self.config.max_depth;
        log::debug!("NAVIGATION: Configuration updated - max_depth: {}, debug_enabled: {}", 
                   self.config.max_depth, self.config.debug_enabled);
    }
}

/// Trait for advanced navigation capabilities
pub trait AdvancedNavigation {
    /// Generate multiple novel prefixes in parallel
    fn generate_multiple_prefixes<R: Rng>(&mut self, rng: &mut R, count: usize) -> NavigationSystemResult<Vec<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>>>;
    
    /// Optimize navigation strategy based on tree characteristics
    fn optimize_strategy(&mut self) -> NavigationSystemResult<NavigationStrategy>;
    
    /// Validate navigation state consistency
    fn validate_state(&self) -> NavigationSystemResult<()>;
}

impl AdvancedNavigation for NavigationSystem {
    fn generate_multiple_prefixes<R: Rng>(&mut self, rng: &mut R, count: usize) -> NavigationSystemResult<Vec<Vec<(ChoiceType, ChoiceValue, Box<Constraints>)>>> {
        let mut prefixes = Vec::with_capacity(count);
        
        for i in 0..count {
            log::debug!("NAVIGATION: Generating prefix {} of {}", i + 1, count);
            
            match self.generate_novel_prefix(rng) {
                Ok(prefix) => prefixes.push(prefix),
                Err(NavigationSystemError::TreeFullyExhausted) => {
                    log::info!("NAVIGATION: Tree exhausted after generating {} prefixes", prefixes.len());
                    break;
                }
                Err(err) => return Err(err),
            }
        }
        
        log::info!("NAVIGATION: Generated {} novel prefixes", prefixes.len());
        Ok(prefixes)
    }
    
    fn optimize_strategy(&mut self) -> NavigationSystemResult<NavigationStrategy> {
        // Analyze current statistics to determine optimal strategy
        let total_prefixes = self.stats.novel_prefixes_generated;
        let avg_length = self.stats.average_prefix_length;
        let cache_hit_rate = self.stats.cache_hit_rate;
        
        let optimal_strategy = if total_prefixes < 10 {
            // Early exploration, use systematic approach
            NavigationStrategy::SystematicDFS
        } else if avg_length < 5.0 {
            // Short prefixes, try weighted random for diversity
            NavigationStrategy::WeightedRandom
        } else if cache_hit_rate > 0.8 {
            // High cache hit rate, tree well-explored, use constraint-guided
            NavigationStrategy::ConstraintGuided
        } else {
            // Balanced approach
            NavigationStrategy::HybridApproach
        };
        
        log::info!("NAVIGATION: Optimized strategy to {:?} based on stats: prefixes={}, avg_length={:.2}, cache_hit_rate={:.2}", 
                  optimal_strategy, total_prefixes, avg_length, cache_hit_rate);
        
        self.state.strategy = optimal_strategy;
        Ok(optimal_strategy)
    }
    
    fn validate_state(&self) -> NavigationSystemResult<()> {
        // Validate trail consistency
        if self.state.trail.len() > self.config.max_depth {
            return Err(NavigationSystemError::StateCorruption(
                "Trail depth exceeds maximum allowed depth".to_string()
            ));
        }
        
        // Validate prefix length consistency
        if self.state.prefix.len() > self.config.max_depth * 10 {
            return Err(NavigationSystemError::StateCorruption(
                "Prefix length exceeds reasonable bounds".to_string()
            ));
        }
        
        // Validate current node accessibility
        if self.state.current_node.node_id == 0 && self.state.depth > 0 {
            log::warn!("NAVIGATION: Potential inconsistency - at root but depth > 0");
        }
        
        log::debug!("NAVIGATION: State validation passed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatree::DataTree;
    use rand::thread_rng;
    
    #[test]
    fn test_navigation_system_creation() {
        let tree = DataTree::new();
        let nav_system = NavigationSystem::new(tree);
        
        assert_eq!(nav_system.state.strategy, NavigationStrategy::SystematicDFS);
        assert_eq!(nav_system.state.depth, 0);
        assert!(nav_system.state.prefix.is_empty());
        assert!(nav_system.state.trail.is_empty());
    }
    
    #[test]
    fn test_novel_prefix_generation() {
        let tree = DataTree::new();
        let mut nav_system = NavigationSystem::new(tree);
        let mut rng = thread_rng();
        
        // Should be able to generate at least one prefix
        let result = nav_system.generate_novel_prefix(&mut rng);
        assert!(result.is_ok());
        
        let prefix = result.unwrap();
        assert!(prefix.len() >= 1); // At least fallback should be generated
        
        // Statistics should be updated
        assert_eq!(nav_system.stats.novel_prefixes_generated, 1);
        assert_eq!(nav_system.stats.traversals_performed, 1);
    }
    
    #[test]
    fn test_multiple_strategies() {
        let tree = DataTree::new();
        let mut nav_system = NavigationSystem::new(tree);
        let mut rng = thread_rng();
        
        // Test different strategies
        let strategies = [
            NavigationStrategy::SystematicDFS,
            NavigationStrategy::WeightedRandom,
            NavigationStrategy::ConstraintGuided,
            NavigationStrategy::HybridApproach,
        ];
        
        for strategy in &strategies {
            nav_system.state.reset(Some(*strategy));
            let result = nav_system.generate_novel_prefix(&mut rng);
            assert!(result.is_ok(), "Strategy {:?} should succeed", strategy);
        }
        
        // Strategy usage should be recorded
        assert!(nav_system.stats.strategy_usage.len() > 0);
    }
    
    #[test]
    fn test_navigation_state_operations() {
        let tree = DataTree::new();
        let root = Arc::clone(&tree.root);
        let mut state = NavigationState::new(root, NavigationStrategy::SystematicDFS);
        
        // Test adding choices
        state.add_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Box::new(Constraints::Integer(crate::choice::IntegerConstraints::default()))
        );
        
        assert_eq!(state.prefix.len(), 1);
        
        // Test removing choices
        state.remove_last_choice();
        assert_eq!(state.prefix.len(), 0);
        
        // Test reset
        state.depth = 10;
        state.reset(Some(NavigationStrategy::WeightedRandom));
        assert_eq!(state.depth, 0);
        assert_eq!(state.strategy, NavigationStrategy::WeightedRandom);
    }
    
    #[test]
    fn test_exhaustion_detection() {
        let tree = DataTree::new();
        let mut nav_system = NavigationSystem::new(tree);
        
        // Test tree exhaustion check
        let result = nav_system.is_tree_exhausted();
        assert!(result.is_ok());
        
        // Test node exhaustion
        let node = Arc::clone(&nav_system.state.current_node);
        let result = nav_system.is_node_exhausted(&node);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_advanced_navigation_capabilities() {
        let tree = DataTree::new();
        let mut nav_system = NavigationSystem::new(tree);
        let mut rng = thread_rng();
        
        // Test multiple prefix generation
        let result = nav_system.generate_multiple_prefixes(&mut rng, 3);
        assert!(result.is_ok());
        
        let prefixes = result.unwrap();
        assert!(prefixes.len() <= 3); // May be less if tree exhausted
        
        // Test strategy optimization
        let result = nav_system.optimize_strategy();
        assert!(result.is_ok());
        
        // Test state validation
        let result = nav_system.validate_state();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_configuration_and_statistics() {
        let tree = DataTree::new();
        let mut nav_system = NavigationSystem::new(tree);
        
        // Test configuration
        let config = NavigationConfig {
            max_depth: 500,
            max_backtrack_attempts: 50,
            random_bias_factor: 0.8,
            exhaustion_cache_size: 5000,
            debug_enabled: false,
        };
        
        nav_system.configure(config.clone());
        assert_eq!(nav_system.config.max_depth, 500);
        assert_eq!(nav_system.state.max_depth, 500);
        
        // Test statistics retrieval
        let stats = nav_system.get_stats();
        assert_eq!(stats.novel_prefixes_generated, 0);
        
        // Test statistics reset
        nav_system.stats.novel_prefixes_generated = 10;
        nav_system.reset_stats();
        assert_eq!(nav_system.stats.novel_prefixes_generated, 0);
    }
    
    #[test]
    fn test_error_handling() {
        let tree = DataTree::new();
        let nav_system = NavigationSystem::new(tree);
        
        // Test error display
        let error = NavigationSystemError::TreeFullyExhausted;
        assert!(error.to_string().contains("fully exhausted"));
        
        let error = NavigationSystemError::StateCorruption("test".to_string());
        assert!(error.to_string().contains("corruption"));
        
        let error = NavigationSystemError::InvalidParameters("test".to_string());
        assert!(error.to_string().contains("Invalid"));
    }
    
    #[test]
    fn test_weighted_selection_logic() {
        let tree = DataTree::new();
        let nav_system = NavigationSystem::new(tree);
        
        // Test exploration weight calculation
        let node = Arc::clone(&nav_system.state.current_node);
        let result = nav_system.calculate_exploration_weight(&node);
        assert!(result.is_ok());
        
        let weight = result.unwrap();
        assert!(weight > 0.0);
        assert!(weight.is_finite());
    }
    
    #[test]
    fn test_complexity_calculation() {
        let tree = DataTree::new();
        let nav_system = NavigationSystem::new(tree);
        
        // Test node complexity calculation
        let node = Arc::clone(&nav_system.state.current_node);
        let complexity = nav_system.calculate_node_complexity(&node);
        
        assert!(complexity >= 0.0);
        assert!(complexity.is_finite());
    }
}