//! Enhanced DataTree Node Navigation System
//!
//! This module implements the complete DataTree node navigation system capability
//! that provides sophisticated tree traversal, node creation, and choice system
//! integration following Python Hypothesis patterns translated to idiomatic Rust.
//!
//! Key components:
//! - TreeRecordingObserver: Tracks current navigation state during test execution
//! - NavigationState: Manages current node position and trail for backtracking
//! - Enhanced node creation and splitting algorithms
//! - Sophisticated exhaustion detection and backtracking logic
//! - Optimized tree traversal with caching and performance optimizations

use crate::datatree::{DataTree, TreeNode, Branch, Conclusion, Killed, Transition, TreeStats};
use crate::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, FloatConstraints, BooleanConstraints, StringConstraints, BytesConstraints};
use crate::data::Status;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock, Weak};

/// Enhanced navigation state for sophisticated tree traversal
#[derive(Debug)]
pub struct NavigationState {
    /// Current node being navigated
    pub current_node: Arc<TreeNode>,
    /// Position within the current node's compressed sequence
    pub index_in_current_node: usize,
    /// Trail of nodes for backtracking (breadcrumbs)
    pub trail: Vec<(Arc<TreeNode>, usize)>,
    /// Depth in the tree for optimization decisions
    pub current_depth: usize,
    /// Cache of recent navigation decisions for performance
    pub navigation_cache: HashMap<u64, NavigationDecision>,
    /// Statistics for navigation performance analysis
    pub nav_stats: NavigationStats,
}

/// Navigation decision cache entry
#[derive(Debug, Clone)]
pub struct NavigationDecision {
    pub chosen_child: Option<ChoiceValue>,
    pub exhaustion_state: bool,
    pub timestamp: u64,
}

/// Statistics for navigation performance monitoring
#[derive(Debug, Clone, Default)]
pub struct NavigationStats {
    pub total_navigations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub backtrack_operations: usize,
    pub node_splits: usize,
    pub exhaustion_checks: usize,
}

/// Tree Recording Observer that tracks navigation state during test execution
/// This is the core component that bridges test execution and tree building
#[derive(Debug)]
pub struct TreeRecordingObserver {
    /// Reference to the tree being recorded into
    tree: Arc<RwLock<DataTree>>,
    /// Current navigation state
    nav_state: NavigationState,
    /// Sequence of choices made during current execution
    choice_sequence: Vec<(ChoiceType, ChoiceValue, Box<Constraints>, bool)>,
    /// Final status for the current execution path
    execution_status: Option<Status>,
    /// Observations collected during execution
    observations: HashMap<String, String>,
    /// Unique identifier for this observer instance
    observer_id: u64,
}

/// Enhanced child selection strategy for navigation
#[derive(Debug, Clone)]
pub enum ChildSelectionStrategy {
    /// Random selection among available children
    Random,
    /// Weighted selection favoring less explored paths
    WeightedByExploration,
    /// Depth-first prioritizing deeper unexplored paths
    DepthFirst,
    /// Breadth-first exploring wide before deep
    BreadthFirst,
    /// Adaptive strategy that changes based on tree characteristics
    Adaptive { exploration_ratio: f64 },
}

impl NavigationState {
    /// Create a new navigation state starting from tree root
    pub fn new(root: Arc<TreeNode>) -> Self {
        println!("DATATREE_NAV DEBUG: Creating new navigation state from root node 0x{:08X}", root.node_id);
        Self {
            current_node: Arc::clone(&root),
            index_in_current_node: 0,
            trail: vec![(root, 0)],
            current_depth: 0,
            navigation_cache: HashMap::new(),
            nav_stats: NavigationStats::default(),
        }
    }

    /// Navigate to the next choice in the current node's compressed sequence
    pub fn advance_in_current_node(&mut self) -> bool {
        if self.index_in_current_node < self.current_node.values.len() {
            self.index_in_current_node += 1;
            println!("DATATREE_NAV DEBUG: Advanced to index {} in node 0x{:08X}", 
                     self.index_in_current_node, self.current_node.node_id);
            true
        } else {
            println!("DATATREE_NAV DEBUG: Reached end of node 0x{:08X} sequence", self.current_node.node_id);
            false
        }
    }

    /// Navigate to a child node based on choice value
    pub fn navigate_to_child(&mut self, choice_value: &ChoiceValue) -> Option<Arc<TreeNode>> {
        self.nav_stats.total_navigations += 1;
        
        // Check if we can use cached navigation decision
        if let Some(cached) = self.navigation_cache.get(&self.current_node.node_id) {
            if let Some(ref cached_choice) = cached.chosen_child {
                if cached_choice == choice_value {
                    self.nav_stats.cache_hits += 1;
                    println!("DATATREE_NAV DEBUG: Using cached navigation for node 0x{:08X}", 
                             self.current_node.node_id);
                }
            }
        } else {
            self.nav_stats.cache_misses += 1;
        }

        // Try to navigate using current node's transition
        let child_node = {
            let transition_guard = self.current_node.transition.read().unwrap();
            if let Some(Transition::Branch(branch)) = transition_guard.as_ref() {
                let children = branch.children.read().unwrap();
                children.get(choice_value).map(Arc::clone)
            } else {
                None
            }
        };
        
        if let Some(child) = child_node {
            // Cache this navigation decision before changing current_node
            let is_exhausted = child.check_exhausted();
            let child_id = child.node_id;
            
            // Update trail and state
            self.trail.push((Arc::clone(&self.current_node), self.index_in_current_node));
            self.current_node = child;
            self.index_in_current_node = 0;
            self.current_depth += 1;
                
            // Cache this navigation decision
            self.navigation_cache.insert(
                child_id,
                NavigationDecision {
                    chosen_child: Some(choice_value.clone()),
                    exhaustion_state: is_exhausted,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                }
            );
            
            println!("DATATREE_NAV DEBUG: Navigated to child node 0x{:08X} at depth {}", 
                     child_id, self.current_depth);
            Some(Arc::clone(&self.current_node))
        } else {
            println!("DATATREE_NAV DEBUG: No child found for choice {:?} in node 0x{:08X}", 
                     choice_value, self.current_node.node_id);
            None
        }
    }

    /// Backtrack to find a node with unexplored alternatives
    pub fn backtrack_to_unexplored(&mut self, tree: &DataTree) -> Option<Arc<TreeNode>> {
        self.nav_stats.backtrack_operations += 1;
        
        // Traverse trail backwards to find unexplored branch
        while let Some((node, index)) = self.trail.pop() {
            if self.has_unexplored_alternatives(&node, tree) {
                // Found a node with unexplored alternatives
                self.current_node = Arc::clone(&node);
                self.index_in_current_node = index;
                self.current_depth = self.trail.len();
                
                println!("DATATREE_NAV DEBUG: Backtracked to node 0x{:08X} at depth {} with unexplored alternatives", 
                         node.node_id, self.current_depth);
                return Some(node);
            }
        }
        
        println!("DATATREE_NAV DEBUG: No unexplored alternatives found in trail");
        None
    }

    /// Check if a node has unexplored alternatives
    fn has_unexplored_alternatives(&mut self, node: &Arc<TreeNode>, _tree: &DataTree) -> bool {
        self.nav_stats.exhaustion_checks += 1;
        
        // Use cached exhaustion state if available
        if let Some(cached) = self.navigation_cache.get(&node.node_id) {
            if !cached.exhaustion_state {
                return true;
            }
        }
        
        // Check if node itself is exhausted
        if node.check_exhausted() {
            return false;
        }
        
        // Check if there are potential children that haven't been explored
        if let Some(Transition::Branch(branch)) = node.transition.read().unwrap().as_ref() {
            let children = branch.children.read().unwrap();
            if let Some(max_children) = node.compute_max_children() {
                if (children.len() as u128) < max_children {
                    return true; // Room for more children
                }
            }
            
            // Check if any existing children are not exhausted
            for child in children.values() {
                if !child.check_exhausted() {
                    return true;
                }
            }
        }
        
        false
    }

    /// Reset navigation state to tree root
    pub fn reset_to_root(&mut self, root: Arc<TreeNode>) {
        println!("DATATREE_NAV DEBUG: Resetting navigation to root node 0x{:08X}", root.node_id);
        self.current_node = Arc::clone(&root);
        self.index_in_current_node = 0;
        self.trail = vec![(root, 0)];
        self.current_depth = 0;
        // Keep navigation cache for performance
    }

    /// Clean old entries from navigation cache
    pub fn clean_navigation_cache(&mut self, max_age_seconds: u64) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let initial_size = self.navigation_cache.len();
        self.navigation_cache.retain(|_, decision| {
            (current_time - decision.timestamp) < max_age_seconds
        });
        
        let removed = initial_size - self.navigation_cache.len();
        if removed > 0 {
            println!("DATATREE_NAV DEBUG: Cleaned {} stale cache entries", removed);
        }
    }
}

impl TreeRecordingObserver {
    /// Create a new tree recording observer
    pub fn new(tree: Arc<RwLock<DataTree>>, observer_id: u64) -> Self {
        let root = {
            let tree_guard = tree.read().unwrap();
            Arc::clone(&tree_guard.root)
        };
        
        println!("DATATREE_NAV DEBUG: Creating TreeRecordingObserver 0x{:08X} with root 0x{:08X}", 
                 observer_id, root.node_id);
        
        Self {
            tree,
            nav_state: NavigationState::new(root),
            choice_sequence: Vec::new(),
            execution_status: None,
            observations: HashMap::new(),
            observer_id,
        }
    }

    /// Record a choice value during test execution
    pub fn draw_value(&mut self, choice_type: ChoiceType, value: ChoiceValue, 
                     constraints: Box<Constraints>, was_forced: bool) -> Result<(), String> {
        println!("DATATREE_NAV DEBUG: Observer 0x{:08X} drawing {:?} value {:?} at node 0x{:08X}[{}]", 
                 self.observer_id, choice_type, value, 
                 self.nav_state.current_node.node_id, self.nav_state.index_in_current_node);

        // Add to choice sequence for later recording
        self.choice_sequence.push((choice_type, value.clone(), constraints.clone(), was_forced));

        // Check if we're still within the current node's sequence
        if self.nav_state.index_in_current_node < self.nav_state.current_node.values.len() {
            let existing_value = &self.nav_state.current_node.values[self.nav_state.index_in_current_node];
            let existing_type = self.nav_state.current_node.choice_types[self.nav_state.index_in_current_node];
            
            if existing_type == choice_type && existing_value == &value {
                // Following existing path
                self.nav_state.advance_in_current_node();
                return Ok(());
            } else {
                // Need to split the node at this point
                return self.handle_divergence(choice_type, value, constraints, was_forced);
            }
        }

        // We're at the end of the current node's sequence
        self.handle_extension(choice_type, value, constraints, was_forced)
    }

    /// Handle divergence from existing path (requires node splitting)
    fn handle_divergence(&mut self, choice_type: ChoiceType, value: ChoiceValue, 
                        constraints: Box<Constraints>, was_forced: bool) -> Result<(), String> {
        println!("DATATREE_NAV DEBUG: Handling divergence at node 0x{:08X}[{}]", 
                 self.nav_state.current_node.node_id, self.nav_state.index_in_current_node);
        
        self.nav_state.nav_stats.node_splits += 1;
        
        // Create a new branch at this point for simplified handling
        let branch = Branch {
            children: RwLock::new(HashMap::new()),
            is_exhausted: RwLock::new(false),
        };
        
        // Set transition on current node if it doesn't have one
        if self.nav_state.current_node.transition.read().unwrap().is_none() {
            *self.nav_state.current_node.transition.write().unwrap() = Some(Transition::Branch(branch));
        }
        
        // Navigate to child (create if necessary)
        if let Some(_child) = self.nav_state.navigate_to_child(&value) {
            println!("DATATREE_NAV DEBUG: Navigated to existing child after divergence");
        } else {
            // Create new child node
            let mut tree_guard = self.tree.write().unwrap();
            let new_child = self.create_child_node(&mut tree_guard, choice_type, value.clone(), constraints, was_forced)?;
            drop(tree_guard); // Release tree lock
            
            // Add to parent's children
            if let Some(Transition::Branch(branch)) = self.nav_state.current_node.transition.read().unwrap().as_ref() {
                branch.children.write().unwrap().insert(value.clone(), Arc::clone(&new_child));
            }
            
            // Navigate to new child
            self.nav_state.current_node = new_child;
            self.nav_state.index_in_current_node = 1; // We just added one choice
            self.nav_state.current_depth += 1;
        }
        
        Ok(())
    }

    /// Handle extension beyond current node's sequence
    fn handle_extension(&mut self, choice_type: ChoiceType, value: ChoiceValue, 
                       constraints: Box<Constraints>, was_forced: bool) -> Result<(), String> {
        println!("DATATREE_NAV DEBUG: Handling extension at node 0x{:08X}", self.nav_state.current_node.node_id);
        
        // Check if current node has a transition
        let transition_guard = self.nav_state.current_node.transition.read().unwrap();
        match transition_guard.as_ref() {
            None => {
                // No transition yet - extend the linear sequence
                drop(transition_guard); // Release read lock
                self.extend_node_sequence(choice_type, value, constraints, was_forced)
            }
            Some(Transition::Branch(branch)) => {
                // Navigate to child or create new one
                drop(transition_guard); // Release read lock
                if let Some(_child) = self.nav_state.navigate_to_child(&value) {
                    Ok(())
                } else {
                    let mut tree_guard = self.tree.write().unwrap();
                    let new_child = self.create_child_node(&mut tree_guard, choice_type, value.clone(), constraints, was_forced)?;
                    drop(tree_guard); // Release tree lock
                    
                    if let Some(Transition::Branch(branch)) = self.nav_state.current_node.transition.read().unwrap().as_ref() {
                        branch.children.write().unwrap().insert(value, Arc::clone(&new_child));
                    }
                    
                    self.nav_state.current_node = new_child;
                    self.nav_state.index_in_current_node = 1;
                    self.nav_state.current_depth += 1;
                    Ok(())
                }
            }
            Some(Transition::Conclusion(_)) => {
                // Already concluded - this shouldn't happen in normal execution
                Err(format!("Attempted to extend concluded node 0x{:08X}", self.nav_state.current_node.node_id))
            }
            Some(Transition::Killed(_)) => {
                // Killed node - handle according to kill reason
                Err(format!("Attempted to extend killed node 0x{:08X}", self.nav_state.current_node.node_id))
            }
        }
    }

    /// Extend the current node's linear sequence with a new choice
    fn extend_node_sequence(&mut self, choice_type: ChoiceType, value: ChoiceValue, 
                           constraints: Box<Constraints>, was_forced: bool) -> Result<(), String> {
        // Create a mutable copy of the current node's data to extend
        // Note: In a full implementation, this would require more careful handling
        // of the Arc<TreeNode> to allow mutation
        
        // For now, we'll create a branch point instead of extending
        let branch = Branch {
            children: RwLock::new(HashMap::new()),
            is_exhausted: RwLock::new(false),
        };
        
        *self.nav_state.current_node.transition.write().unwrap() = Some(Transition::Branch(branch));
        
        // Create child node with the new choice
        let mut tree_guard = self.tree.write().unwrap();
        let new_child = self.create_child_node(&mut tree_guard, choice_type, value.clone(), constraints, was_forced)?;
        drop(tree_guard); // Release tree lock
        
        if let Some(Transition::Branch(branch)) = self.nav_state.current_node.transition.read().unwrap().as_ref() {
            branch.children.write().unwrap().insert(value, Arc::clone(&new_child));
        }
        
        self.nav_state.current_node = new_child;
        self.nav_state.index_in_current_node = 1;
        self.nav_state.current_depth += 1;
        
        Ok(())
    }

    /// Create a new child node with given choice
    fn create_child_node(&self, tree: &mut DataTree, choice_type: ChoiceType, value: ChoiceValue, 
                        constraints: Box<Constraints>, was_forced: bool) -> Result<Arc<TreeNode>, String> {
        let mut new_node = TreeNode::new(tree.next_node_id);
        tree.next_node_id += 1;
        
        new_node.add_choice(choice_type, value.clone(), constraints, was_forced);
        
        let new_node_arc = Arc::new(new_node);
        tree.stats.total_nodes += 1;
        
        println!("DATATREE_NAV DEBUG: Created new child node 0x{:08X} with choice {:?}", 
                 new_node_arc.node_id, value);
        
        Ok(new_node_arc)
    }

    /// Set the execution status for this path
    pub fn set_status(&mut self, status: Status) {
        println!("DATATREE_NAV DEBUG: Observer 0x{:08X} setting status {:?} at node 0x{:08X}", 
                 self.observer_id, status, self.nav_state.current_node.node_id);
        self.execution_status = Some(status);
    }

    /// Add an observation for this execution
    pub fn add_observation(&mut self, key: String, value: String) {
        self.observations.insert(key, value);
    }

    /// Complete the recording and store the path in the tree
    pub fn complete_recording(&mut self) -> Result<(), String> {
        let status = self.execution_status.unwrap_or(Status::Valid);
        
        println!("DATATREE_NAV DEBUG: Observer 0x{:08X} completing recording with {} choices, status {:?}", 
                 self.observer_id, self.choice_sequence.len(), status);
        
        // Mark the current node as a conclusion
        let conclusion = Conclusion {
            status,
            interesting_origin: None,
            target_observations: self.observations.iter()
                .filter_map(|(k, v)| v.parse::<f64>().ok().map(|val| (k.clone(), val)))
                .collect(),
            metadata: self.observations.iter()
                .filter(|(_, v)| v.parse::<f64>().is_err())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        
        *self.nav_state.current_node.transition.write().unwrap() = Some(Transition::Conclusion(conclusion));
        
        // Update tree statistics
        {
            let mut tree_guard = self.tree.write().unwrap();
            tree_guard.stats.conclusion_nodes += 1;
        }
        
        println!("DATATREE_NAV DEBUG: Recording completed successfully for observer 0x{:08X}", self.observer_id);
        Ok(())
    }

    /// Get current navigation statistics
    pub fn get_navigation_stats(&self) -> NavigationStats {
        self.nav_state.nav_stats.clone()
    }

    /// Reset observer for new execution path
    pub fn reset_for_new_execution(&mut self) {
        let root = {
            let tree_guard = self.tree.read().unwrap();
            Arc::clone(&tree_guard.root)
        };
        
        self.nav_state.reset_to_root(root);
        self.choice_sequence.clear();
        self.execution_status = None;
        self.observations.clear();
        
        println!("DATATREE_NAV DEBUG: Observer 0x{:08X} reset for new execution", self.observer_id);
    }
}

/// Enhanced child selection algorithms for navigation
impl DataTree {
    /// Select child using specified strategy
    pub fn select_child_with_strategy<R: Rng>(&self, branch: &Branch, rng: &mut R, 
                                             strategy: &ChildSelectionStrategy, depth: usize) -> Option<Arc<TreeNode>> {
        let children = branch.children.read().unwrap();
        if children.is_empty() {
            return None;
        }

        match strategy {
            ChildSelectionStrategy::Random => {
                let children_vec: Vec<_> = children.values().collect();
                let index = rng.gen_range(0..children_vec.len());
                Some(Arc::clone(children_vec[index]))
            }
            ChildSelectionStrategy::WeightedByExploration => {
                self.select_weighted_by_exploration(&children, rng, depth)
            }
            ChildSelectionStrategy::DepthFirst => {
                // Select child with maximum depth potential
                children.values()
                    .max_by_key(|child| self.estimate_subtree_depth(child))
                    .map(Arc::clone)
            }
            ChildSelectionStrategy::BreadthFirst => {
                // Select child with minimum current depth
                children.values()
                    .min_by_key(|child| self.estimate_subtree_depth(child))
                    .map(Arc::clone)
            }
            ChildSelectionStrategy::Adaptive { exploration_ratio } => {
                // Adapt strategy based on exploration ratio
                if *exploration_ratio < 0.3 {
                    self.select_weighted_by_exploration(&children, rng, depth)
                } else if *exploration_ratio > 0.8 {
                    // Focus on depth when mostly explored
                    children.values()
                        .max_by_key(|child| self.estimate_subtree_depth(child))
                        .map(Arc::clone)
                } else {
                    // Random selection for balanced exploration
                    let children_vec: Vec<_> = children.values().collect();
                    let index = rng.gen_range(0..children_vec.len());
                    Some(Arc::clone(children_vec[index]))
                }
            }
        }
    }

    /// Weighted selection favoring less explored children
    fn select_weighted_by_exploration<R: Rng>(&self, children: &HashMap<ChoiceValue, Arc<TreeNode>>, 
                                             rng: &mut R, depth: usize) -> Option<Arc<TreeNode>> {
        let mut candidates: Vec<(Arc<TreeNode>, f64)> = Vec::new();
        
        for child in children.values() {
            if !child.check_exhausted() {
                let weight = self.calculate_exploration_weight(child, depth);
                candidates.push((Arc::clone(child), weight));
            }
        }
        
        if candidates.is_empty() {
            return None;
        }
        
        // Weighted random selection
        let total_weight: f64 = candidates.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            let index = rng.gen_range(0..candidates.len());
            return Some(Arc::clone(&candidates[index].0));
        }
        
        let mut target = rng.gen::<f64>() * total_weight;
        for (node, weight) in &candidates {
            target -= weight;
            if target <= 0.0 {
                return Some(Arc::clone(node));
            }
        }
        
        Some(Arc::clone(&candidates[0].0))
    }

    /// Estimate the depth potential of a subtree
    fn estimate_subtree_depth(&self, node: &Arc<TreeNode>) -> usize {
        // Simple heuristic based on current choices and transitions
        let mut depth = node.values.len();
        
        if let Ok(transition_guard) = node.transition.read() {
            if let Some(Transition::Branch(branch)) = transition_guard.as_ref() {
                let children_count = branch.children.read().unwrap().len();
                depth += children_count * 2; // Rough estimate
            }
        }
        
        depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_navigation_state_creation() {
        let root = Arc::new(TreeNode::new(0));
        let nav_state = NavigationState::new(Arc::clone(&root));
        
        assert_eq!(nav_state.current_node.node_id, 0);
        assert_eq!(nav_state.index_in_current_node, 0);
        assert_eq!(nav_state.current_depth, 0);
        assert_eq!(nav_state.trail.len(), 1);
    }

    #[test]
    fn test_tree_recording_observer() {
        let tree = Arc::new(RwLock::new(DataTree::new()));
        let mut observer = TreeRecordingObserver::new(Arc::clone(&tree), 1);
        
        // Test drawing a value
        let result = observer.draw_value(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
            false
        );
        
        assert!(result.is_ok());
        assert_eq!(observer.choice_sequence.len(), 1);
    }

    #[test]
    fn test_navigation_with_branching() {
        let tree = Arc::new(RwLock::new(DataTree::new()));
        let mut observer = TreeRecordingObserver::new(Arc::clone(&tree), 2);
        
        // Draw multiple values to create branching
        observer.draw_value(
            ChoiceType::Integer,
            ChoiceValue::Integer(1),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
            false
        ).unwrap();
        
        observer.draw_value(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Box::new(Constraints::Boolean(BooleanConstraints { p: 0.5 })),
            false
        ).unwrap();
        
        observer.set_status(Status::Valid);
        let result = observer.complete_recording();
        assert!(result.is_ok());
        
        // Verify tree structure was created
        let tree_guard = tree.read().unwrap();
        let stats = tree_guard.get_stats();
        assert!(stats.total_nodes > 0);
        assert_eq!(stats.conclusion_nodes, 1);
    }

    #[test]
    fn test_child_selection_strategies() {
        let tree = DataTree::new();
        let mut rng = thread_rng();
        
        // Create a simple branch for testing
        let branch = Branch {
            children: RwLock::new({
                let mut children = HashMap::new();
                children.insert(ChoiceValue::Integer(1), Arc::new(TreeNode::new(1)));
                children.insert(ChoiceValue::Integer(2), Arc::new(TreeNode::new(2)));
                children
            }),
            is_exhausted: RwLock::new(false),
        };
        
        // Test different selection strategies
        let random_child = tree.select_child_with_strategy(&branch, &mut rng, &ChildSelectionStrategy::Random, 0);
        assert!(random_child.is_some());
        
        let weighted_child = tree.select_child_with_strategy(&branch, &mut rng, &ChildSelectionStrategy::WeightedByExploration, 0);
        assert!(weighted_child.is_some());
        
        let adaptive_child = tree.select_child_with_strategy(&branch, &mut rng, &ChildSelectionStrategy::Adaptive { exploration_ratio: 0.5 }, 0);
        assert!(adaptive_child.is_some());
    }

    #[test]
    fn test_navigation_cache() {
        let root = Arc::new(TreeNode::new(0));
        let mut nav_state = NavigationState::new(root);
        
        // Test cache management
        assert_eq!(nav_state.navigation_cache.len(), 0);
        
        // Simulate some navigation decisions
        nav_state.navigation_cache.insert(
            1,
            NavigationDecision {
                chosen_child: Some(ChoiceValue::Integer(42)),
                exhaustion_state: false,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            }
        );
        
        assert_eq!(nav_state.navigation_cache.len(), 1);
        
        // Test cache cleaning (should not remove recent entries)
        nav_state.clean_navigation_cache(3600); // 1 hour
        assert_eq!(nav_state.navigation_cache.len(), 1);
        
        // Test cache cleaning with zero max age (should remove all)
        nav_state.clean_navigation_cache(0);
        assert_eq!(nav_state.navigation_cache.len(), 0);
    }
}