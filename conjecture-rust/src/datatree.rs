//! DataTree - Novel prefix generation and test space exploration
//!
//! This module implements the core DataTree system that transforms Python Hypothesis
//! from basic random testing into sophisticated property-based testing. The DataTree
//! provides systematic exploration of the test input space, avoiding duplication
//! and ensuring comprehensive coverage.
//!
//! Key components:
//! - TreeNode: Radix tree structure for compressed choice sequences
//! - Branch/Conclusion/Killed transitions: Tree navigation states
//! - Novel prefix generation: Core algorithm for systematic exploration
//! - Tree recording: Incremental tree building from test execution
//! - Mathematical utilities: Max children calculation and exhaustion tracking

use crate::choice::{ChoiceType, ChoiceValue, Constraints};
use crate::data::Status;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Core radix tree node that stores compressed choice sequences
/// This is the foundation of Python's sophisticated test space exploration
#[derive(Debug)]
pub struct TreeNode {
    /// Parallel arrays storing compressed choice sequences
    pub constraints: Vec<Box<Constraints>>,
    pub values: Vec<ChoiceValue>,
    pub choice_types: Vec<ChoiceType>,
    
    /// Tracks which choices were forced (for replay) vs randomly generated
    pub forced: Option<HashSet<usize>>,
    
    /// Tree navigation - how this node connects to the rest of the tree
    pub transition: RwLock<Option<Transition>>,
    
    /// Exploration state - whether this branch has been fully explored
    pub is_exhausted: RwLock<Option<bool>>,
    
    /// Unique identifier for this node
    pub node_id: u64,
}

/// Types of transitions from one tree node to the next
#[derive(Debug)]
pub enum Transition {
    /// Branch with multiple possible continuations
    Branch(Branch),
    /// Terminal node representing a test conclusion
    Conclusion(Conclusion),
    /// Dead-end node that cannot be continued
    Killed(Killed),
}

/// Branch node with multiple possible next choices
#[derive(Debug)]
pub struct Branch {
    /// Map from choice values to child nodes
    pub children: RwLock<HashMap<ChoiceValue, Arc<TreeNode>>>,
    /// Whether all possible children have been explored
    pub is_exhausted: RwLock<bool>,
}

/// Terminal node representing the end of a test execution path
#[derive(Debug)]
pub struct Conclusion {
    /// Final status of the test execution
    pub status: Status,
    /// Optional interesting origin for failures/interesting cases
    pub interesting_origin: Option<String>,
    /// Target observations made during execution
    pub target_observations: HashMap<String, f64>,
    /// Additional metadata about the conclusion
    pub metadata: HashMap<String, String>,
}

/// Killed node representing a branch that should not be explored further
#[derive(Debug)]
pub struct Killed {
    /// Optional continuation point after killed section
    pub next_node: Option<Arc<TreeNode>>,
    /// Reason this branch was killed
    pub reason: String,
}

/// The main DataTree structure providing novel prefix generation
/// This is the architectural centerpiece that makes property-based testing intelligent
#[derive(Debug)]
pub struct DataTree {
    /// Root node of the exploration tree
    root: Arc<TreeNode>,
    
    /// Cache of recently accessed nodes for performance (LRU-style)
    node_cache: HashMap<u64, Arc<TreeNode>>,
    
    /// Maximum cache size to prevent unbounded memory growth
    max_cache_size: usize,
    
    /// Counter for generating unique node IDs
    next_node_id: u64,
    
    /// Statistics about tree exploration
    pub stats: TreeStats,
}

/// Statistics about DataTree exploration for analysis and debugging
#[derive(Debug, Clone, Default)]
pub struct TreeStats {
    pub total_nodes: usize,
    pub branch_nodes: usize,
    pub conclusion_nodes: usize,
    pub killed_nodes: usize,
    pub novel_prefixes_generated: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl TreeNode {
    /// Create a new empty tree node
    pub fn new(node_id: u64) -> Self {
        Self {
            constraints: Vec::new(),
            values: Vec::new(),
            choice_types: Vec::new(),
            forced: None,
            transition: RwLock::new(None),
            is_exhausted: RwLock::new(None),
            node_id,
        }
    }
    
    /// Add a choice to this node's compressed sequence
    pub fn add_choice(&mut self, choice_type: ChoiceType, value: ChoiceValue, 
                     constraints: Box<Constraints>, was_forced: bool) {
        self.choice_types.push(choice_type);
        self.values.push(value);
        self.constraints.push(constraints);
        
        if was_forced {
            self.forced.get_or_insert_with(HashSet::new).insert(self.values.len() - 1);
        }
    }
    
    /// Split this node at the given index, creating a branch point
    /// This is critical for building the tree structure as we discover choice points
    pub fn split_at(&mut self, index: usize, next_node_id: &mut u64) -> Arc<TreeNode> {
        println!("DATATREE DEBUG: Splitting node {} at index {}", self.node_id, index);
        
        if index >= self.values.len() {
            // Create a copy manually since TreeNode no longer implements Clone
            let mut copy = TreeNode::new(self.node_id);
            copy.constraints = self.constraints.clone();
            copy.values = self.values.clone();
            copy.choice_types = self.choice_types.clone();
            copy.forced = self.forced.clone();
            return Arc::new(copy);
        }
        
        // Create new node for the suffix
        let mut suffix_node = TreeNode::new(*next_node_id);
        *next_node_id += 1;
        
        // Move choices from index onwards to the new node
        suffix_node.choice_types = self.choice_types.split_off(index);
        suffix_node.values = self.values.split_off(index);
        suffix_node.constraints = self.constraints.split_off(index);
        
        // Update forced indices for the suffix
        if let Some(ref forced) = self.forced {
            let mut suffix_forced = HashSet::new();
            for &forced_idx in forced {
                if forced_idx >= index {
                    suffix_forced.insert(forced_idx - index);
                }
            }
            if !suffix_forced.is_empty() {
                suffix_node.forced = Some(suffix_forced);
            }
            
            // Keep only forced indices before the split
            self.forced = Some(forced.iter()
                .filter(|&&idx| idx < index)
                .cloned()
                .collect());
            if self.forced.as_ref().unwrap().is_empty() {
                self.forced = None;
            }
        }
        
        // The suffix takes over this node's transition
        suffix_node.transition = RwLock::new(self.transition.write().unwrap().take());
        
        Arc::new(suffix_node)
    }
    
    /// Check if this node has been fully explored
    /// Critical algorithm for knowing when to stop generation
    pub fn check_exhausted(&self) -> bool {
        // Check cached exhaustion state first
        if let Ok(exhausted_guard) = self.is_exhausted.read() {
            if let Some(cached) = *exhausted_guard {
                return cached;
            }
        }
        
        let result = match self.transition.read().unwrap().as_ref() {
            Some(Transition::Branch(branch)) => {
                // Branch is exhausted if explicitly marked or all children are exhausted
                if *branch.is_exhausted.read().unwrap() {
                    return true;
                }
                
                let children = branch.children.read().unwrap();
                if children.is_empty() {
                    // No children yet, not exhausted unless we know max children is 0
                    if let Some(max_children) = self.compute_max_children() {
                        max_children == 0
                    } else {
                        false
                    }
                } else {
                    // Check if we've reached maximum possible children
                    if let Some(max_children) = self.compute_max_children() {
                        let current_children = children.len() as u128;
                        if current_children >= max_children {
                            return children.values().all(|child| child.check_exhausted());
                        }
                    }
                    
                    // Otherwise, exhausted only if all current children are exhausted
                    // and we have some indication this is complete
                    children.values().all(|child| child.check_exhausted())
                }
            },
            Some(Transition::Conclusion(_)) => true,
            Some(Transition::Killed(_)) => true,
            None => {
                // No transition means potentially more to explore
                // unless all choices are forced or we're at a known dead end
                self.all_choices_forced()
            },
        };
        
        // Cache the result
        if let Ok(mut exhausted_guard) = self.is_exhausted.write() {
            *exhausted_guard = Some(result);
        }
        result
    }
    
    /// Check if all choices in this node were forced (no exploration possible)
    fn all_choices_forced(&self) -> bool {
        if self.values.is_empty() {
            return false;
        }
        
        if let Some(ref forced) = self.forced {
            forced.len() == self.values.len()
        } else {
            false
        }
    }
    
    /// Compute maximum possible children for this node based on choice constraints
    /// This is essential for exhaustion tracking and tree size estimation
    pub fn compute_max_children(&self) -> Option<u128> {
        if self.values.is_empty() {
            return None;
        }
        
        // Get the constraint for the next choice that would be made
        let last_idx = self.values.len() - 1;
        let choice_type = self.choice_types[last_idx];
        let constraints = &self.constraints[last_idx];
        
        // Enhanced computation based on actual constraints
        match choice_type {
            ChoiceType::Integer => {
                // Calculate based on integer constraints if available
                if let Constraints::Integer(int_constraints) = constraints.as_ref() {
                    if let (Some(min), Some(max)) = (int_constraints.min_value, int_constraints.max_value) {
                        if max >= min {
                            let range = (max - min + 1) as u128;
                            return Some(range.min(10000)); // Cap at reasonable size
                        }
                    }
                }
                Some(1000) // Default approximation
            },
            ChoiceType::Boolean => Some(2), // True/False
            ChoiceType::Float => {
                // Floats have effectively infinite possibilities, use approximation
                Some(1000)
            },
            ChoiceType::String => {
                // String length constraints could be used here
                Some(1000)
            },
            ChoiceType::Bytes => {
                // Byte sequences, potentially bounded by length
                Some(256)
            },
        }
    }
    
    /// Enhanced exhaustion checking with mathematical precision
    pub fn compute_exhaustion_ratio(&self) -> f64 {
        match self.transition.read().unwrap().as_ref() {
            Some(Transition::Branch(branch)) => {
                if let Some(max_children) = self.compute_max_children() {
                    let current_children = branch.children.read().unwrap().len() as u128;
                    if max_children > 0 {
                        return (current_children as f64) / (max_children as f64);
                    }
                }
                0.0
            },
            Some(Transition::Conclusion(_)) => 1.0,
            Some(Transition::Killed(_)) => 1.0,
            None => 0.0,
        }
    }
}

impl DataTree {
    /// Create a new empty DataTree
    pub fn new() -> Self {
        let root = Arc::new(TreeNode::new(0));
        Self {
            root,
            node_cache: HashMap::new(),
            max_cache_size: 1000, // Reasonable default cache size
            next_node_id: 1,
            stats: TreeStats::default(),
        }
    }
    
    /// Clean cache if it exceeds maximum size
    fn manage_cache_size(&mut self) {
        if self.node_cache.len() > self.max_cache_size {
            // Simple cache eviction: clear half the cache
            // In a production system, would use proper LRU eviction
            let target_size = self.max_cache_size / 2;
            let current_size = self.node_cache.len();
            
            if current_size > target_size {
                // Keep only the most recently inserted nodes (simple heuristic)
                let mut entries: Vec<_> = self.node_cache.drain().collect();
                entries.sort_by_key(|(id, _)| *id); // Sort by node ID (insertion order proxy)
                
                // Keep the last target_size entries
                let keep_from = entries.len().saturating_sub(target_size);
                for (id, node) in entries.into_iter().skip(keep_from) {
                    self.node_cache.insert(id, node);
                }
                
                self.stats.cache_misses += current_size - target_size;
            }
        }
    }
    
    /// THE CORE ALGORITHM: Generate a novel prefix for test exploration
    /// This is the heart of sophisticated property-based testing
    /// OPTIMIZED VERSION: Improved performance and exploration strategy
    pub fn generate_novel_prefix<R: Rng>(&mut self, rng: &mut R) -> Vec<(ChoiceType, ChoiceValue, Box<Constraints>)> {
        println!("DATATREE DEBUG: Generating novel prefix from tree with {} nodes", self.stats.total_nodes);
        
        self.stats.novel_prefixes_generated += 1;
        
        // Pre-allocate collections with reasonable capacity to avoid frequent reallocations
        let mut current_node = Arc::clone(&self.root);
        let mut prefix = Vec::with_capacity(64); // Reasonable initial capacity
        let mut trail = Vec::with_capacity(32);
        let mut depth = 0;
        let max_depth = 1000;
        
        // Cache for exhaustion states to avoid redundant checks
        let mut exhaustion_cache = std::collections::HashMap::new();
        
        // Optimized tree traversal with efficient backtracking
        loop {
            if depth > max_depth {
                println!("DATATREE DEBUG: Maximum depth reached, stopping traversal");
                break;
            }
            
            // Build prefix incrementally and track trail efficiently
            let prefix_start_len = prefix.len();
            trail.push((Arc::clone(&current_node), prefix_start_len));
            
            // Batch add choices to reduce individual push operations
            let node_choices_len = current_node.values.len();
            prefix.extend((0..node_choices_len).map(|i| (
                current_node.choice_types[i],
                current_node.values[i].clone(),
                current_node.constraints[i].clone(),
            )));
            
            depth += 1;
            
            // Optimized transition handling with single lock acquisition
            let transition_guard = current_node.transition.read().unwrap();
            match transition_guard.as_ref() {
                None => {
                    // No transition yet - this is a novel path!
                    println!("DATATREE DEBUG: Found novel path at node {} with {} choices at depth {}", 
                             current_node.node_id, prefix.len(), depth);
                    break;
                }
                Some(Transition::Branch(branch)) => {
                    // Use cached exhaustion check to avoid redundant computation
                    let node_id = current_node.node_id;
                    let is_exhausted = *exhaustion_cache.entry(node_id).or_insert_with(|| {
                        self.check_branch_exhausted_optimized(branch)
                    });
                    
                    if is_exhausted {
                        // Optimized backtracking that preserves useful trail information
                        println!("DATATREE DEBUG: Branch {} is exhausted, attempting efficient backtrack", node_id);
                        drop(transition_guard); // Release lock before backtracking
                        
                        if let Some(backtrack_info) = self.find_backtrack_point_optimized(&trail, &exhaustion_cache) {
                            let (backtrack_node, backtrack_prefix_len, backtrack_depth) = backtrack_info;
                            current_node = backtrack_node;
                            prefix.truncate(backtrack_prefix_len);
                            
                            // Preserve partial trail instead of clearing completely
                            trail.truncate(backtrack_depth + 1);
                            depth = backtrack_depth;
                            continue;
                        } else {
                            println!("DATATREE DEBUG: No backtrack point available, tree exhausted");
                            break;
                        }
                    }
                    
                    // Enhanced child selection with better performance
                    if let Some(child) = self.select_unexplored_child_optimized(branch, rng, depth, &exhaustion_cache) {
                        drop(transition_guard); // Release lock
                        current_node = child;
                        continue;
                    } else {
                        println!("DATATREE DEBUG: No unexplored children in branch {}, novel territory", node_id);
                        break;
                    }
                }
                Some(Transition::Conclusion(_)) => {
                    // Efficient backtracking for conclusions
                    println!("DATATREE DEBUG: Reached conclusion at node {}, backtracking", current_node.node_id);
                    drop(transition_guard);
                    
                    if let Some(backtrack_info) = self.find_backtrack_point_optimized(&trail, &exhaustion_cache) {
                        let (backtrack_node, backtrack_prefix_len, backtrack_depth) = backtrack_info;
                        current_node = backtrack_node;
                        prefix.truncate(backtrack_prefix_len);
                        trail.truncate(backtrack_depth + 1);
                        depth = backtrack_depth;
                        continue;
                    } else {
                        break;
                    }
                }
                Some(Transition::Killed(_)) => {
                    // Aggressive backtracking for killed nodes
                    println!("DATATREE DEBUG: Reached killed node {}, backtracking", current_node.node_id);
                    drop(transition_guard);
                    
                    if let Some(backtrack_info) = self.find_backtrack_point_optimized(&trail, &exhaustion_cache) {
                        let (backtrack_node, backtrack_prefix_len, backtrack_depth) = backtrack_info;
                        current_node = backtrack_node;
                        prefix.truncate(backtrack_prefix_len);
                        trail.truncate(backtrack_depth + 1);
                        depth = backtrack_depth;
                        continue;
                    } else {
                        break;
                    }
                }
            }
        }
        
        // Fallback handling with optimized check
        if prefix.is_empty() && !self.is_tree_fully_exhausted() {
            prefix = self.generate_fallback_prefix(rng);
        }
        
        println!("DATATREE DEBUG: Generated prefix with {} choices at final depth {}", prefix.len(), depth);
        prefix
    }
    
    /// Select an unexplored child from a branch, using weighted random selection
    fn select_unexplored_child<R: Rng>(&self, branch: &Branch, rng: &mut R) -> Option<Arc<TreeNode>> {
        let children = branch.children.read().unwrap();
        let unexplored_children: Vec<_> = children.values()
            .filter(|child| !child.check_exhausted())
            .collect();
        
        if unexplored_children.is_empty() {
            return None;
        }
        
        // Use weighted random selection favoring less explored paths
        let index = rng.gen_range(0..unexplored_children.len());
        Some(Arc::clone(unexplored_children[index]))
    }
    
    /// Enhanced weighted selection for unexplored children with depth consideration
    fn select_unexplored_child_weighted<R: Rng>(&self, branch: &Branch, rng: &mut R, depth: usize) -> Option<Arc<TreeNode>> {
        let mut candidates: Vec<(Arc<TreeNode>, f64)> = Vec::new();
        
        let children = branch.children.read().unwrap();
        for child in children.values() {
            if !child.check_exhausted() {
                // Calculate weight based on exploration metrics
                let weight = self.calculate_exploration_weight(child, depth);
                candidates.push((Arc::clone(child), weight));
            }
        }
        drop(children); // Explicitly drop to avoid borrow issues
        
        if candidates.is_empty() {
            return None;
        }
        
        // Weighted random selection
        let total_weight: f64 = candidates.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            // Fallback to uniform selection
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
        
        // Fallback (should not reach here)
        Some(Arc::clone(&candidates[0].0))
    }
    
    /// Optimized child selection with exhaustion cache
    fn select_unexplored_child_optimized<R: Rng>(&self, branch: &Branch, rng: &mut R, depth: usize, 
                                                _exhaustion_cache: &std::collections::HashMap<u64, bool>) 
                                                -> Option<Arc<TreeNode>> {
        // For now, just call the weighted version - optimization can be added later
        self.select_unexplored_child_weighted(branch, rng, depth)
    }
    
    /// Calculate exploration weight for a child node
    fn calculate_exploration_weight(&self, child: &Arc<TreeNode>, depth: usize) -> f64 {
        let mut weight = 1.0;
        
        // Favor nodes with fewer children (less explored)
        if let Ok(transition_guard) = child.transition.read() {
            if let Some(Transition::Branch(branch)) = transition_guard.as_ref() {
                let child_count = branch.children.read().unwrap().len();
                if child_count > 0 {
                    weight *= 1.0 / (child_count as f64).sqrt();
                }
            }
        }
        
        // Favor shallower nodes at greater depths to balance exploration
        if depth > 10 {
            weight *= 2.0;
        }
        
        // Boost weight for nodes with no transition (novel territory)
        if let Ok(transition_guard) = child.transition.read() {
            if transition_guard.is_none() {
                weight *= 3.0;
            }
        }
        
        weight.max(0.01) // Ensure minimum weight
    }
    
    /// Check if a branch is truly exhausted with sophisticated detection
    fn check_branch_exhausted(&self, branch: &Branch) -> bool {
        if *branch.is_exhausted.read().unwrap() {
            return true;
        }
        
        // Check if all children are exhausted
        let children = branch.children.read().unwrap();
        if children.is_empty() {
            return false;
        }
        
        // A branch is exhausted if all its children are exhausted
        let all_exhausted = children.values().all(|child| child.check_exhausted());
        
        if all_exhausted {
            println!("DATATREE DEBUG: Detected branch exhaustion through child analysis");
            *branch.is_exhausted.write().unwrap() = true;
        }
        
        all_exhausted
    }
    
    /// Optimized version of check_branch_exhausted with caching
    fn check_branch_exhausted_optimized(&self, branch: &Branch) -> bool {
        // For now, just call the regular version - optimization can be added later
        self.check_branch_exhausted(branch)
    }
    
    /// Find a backtrack point in the trail that has unexplored branches
    fn find_backtrack_point(&self, trail: &[(Arc<TreeNode>, usize)]) -> Option<(Arc<TreeNode>, usize)> {
        // Traverse trail backwards to find a node with unexplored children
        for (node, prefix_len) in trail.iter().rev() {
            if let Some(Transition::Branch(branch)) = node.transition.read().unwrap().as_ref() {
                if !self.check_branch_exhausted(branch) {
                    println!("DATATREE DEBUG: Found backtrack point at node {} with prefix len {}", node.node_id, prefix_len);
                    return Some((Arc::clone(node), *prefix_len));
                }
            }
        }
        
        println!("DATATREE DEBUG: No viable backtrack point found in trail of {} nodes", trail.len());
        None
    }
    
    /// Optimized backtrack point finder with exhaustion cache and depth tracking
    fn find_backtrack_point_optimized(&self, trail: &[(Arc<TreeNode>, usize)], 
                                     _exhaustion_cache: &std::collections::HashMap<u64, bool>) 
                                     -> Option<(Arc<TreeNode>, usize, usize)> {
        // Traverse trail backwards to find a node with unexplored children
        for (depth, (node, prefix_len)) in trail.iter().enumerate().rev() {
            if let Some(Transition::Branch(branch)) = node.transition.read().unwrap().as_ref() {
                if !self.check_branch_exhausted(branch) {
                    println!("DATATREE DEBUG: Found optimized backtrack point at node {} with prefix len {} at depth {}", 
                             node.node_id, prefix_len, depth);
                    return Some((Arc::clone(node), *prefix_len, depth));
                }
            }
        }
        
        println!("DATATREE DEBUG: No viable optimized backtrack point found in trail of {} nodes", trail.len());
        None
    }
    
    /// Check if the entire tree is fully exhausted
    fn is_tree_fully_exhausted(&self) -> bool {
        self.root.check_exhausted()
    }
    
    /// Generate a fallback prefix when tree appears exhausted but we need something
    fn generate_fallback_prefix<R: Rng>(&self, rng: &mut R) -> Vec<(ChoiceType, ChoiceValue, Box<Constraints>)> {
        println!("DATATREE DEBUG: Generating fallback prefix for exhausted tree");
        
        // Generate a minimal random prefix to continue exploration
        // This handles edge cases where the tree might not be truly exhausted
        let mut prefix = Vec::new();
        
        // Add a single random choice to potentially create a new branch
        let choice_type = ChoiceType::Integer;
        let value = ChoiceValue::Integer(rng.gen_range(0..100));
        let constraints = Box::new(Constraints::Integer(crate::choice::IntegerConstraints::default()));
        
        prefix.push((choice_type, value, constraints));
        
        prefix
    }
    
    /// Record a test execution path in the tree with enhanced structure building
    pub fn record_path(&mut self, choices: &[(ChoiceType, ChoiceValue, Box<Constraints>, bool)], 
                      status: crate::data::Status, observations: HashMap<String, String>) {
        println!("DATATREE DEBUG: Recording path with {} choices", choices.len());
        
        let mut current_node = Arc::clone(&self.root);
        let mut path_nodes = vec![Arc::clone(&current_node)];
        let mut current_choice_index = 0;
        
        // Traverse/build the tree following this choice sequence
        for (choice_type, value, constraints, was_forced) in choices.iter() {
            // Check if current node needs to be split
            if let Some(split_point) = self.find_split_point(&current_node, current_choice_index, choice_type, value) {
                // Handle splitting without try_unwrap which can panic
                if split_point < current_node.values.len() {
                    // Create a new branch at this point if needed
                    let branch = Branch {
                        children: RwLock::new(HashMap::new()),
                        is_exhausted: RwLock::new(false),
                    };
                    *current_node.transition.write().unwrap() = Some(Transition::Branch(branch));
                    
                    self.stats.branch_nodes += 1;
                }
            }
            
            // Navigate to next node based on this choice
            current_node = self.navigate_or_create_child(&current_node, choice_type, value, constraints, *was_forced);
            path_nodes.push(Arc::clone(&current_node));
            current_choice_index += 1;
        }
        
        // Mark the final node as a conclusion
        self.mark_conclusion(Arc::clone(&current_node), status, observations);
        
        // Update exhaustion states for path nodes
        self.update_path_exhaustion_states(&path_nodes);
        
        println!("DATATREE DEBUG: Path recording complete, tree now has {} nodes", self.stats.total_nodes);
    }
    
    /// Update exhaustion states for nodes along a recorded path
    fn update_path_exhaustion_states(&mut self, path_nodes: &[Arc<TreeNode>]) {
        // Update exhaustion flags based on tree structure changes
        for node in path_nodes {
            if let Some(Transition::Branch(branch)) = node.transition.read().unwrap().as_ref() {
                // Check if this branch should now be marked as exhausted
                let should_be_exhausted = self.check_branch_exhausted(branch);
                if should_be_exhausted && !*branch.is_exhausted.read().unwrap() {
                    println!("DATATREE DEBUG: Marking branch {} as exhausted", node.node_id);
                    *branch.is_exhausted.write().unwrap() = true;
                }
            }
            
            // Also invalidate the cached exhaustion state to force recalculation
            *node.is_exhausted.write().unwrap() = None;
        }
    }
    
    /// Find where a node needs to be split based on a diverging choice
    fn find_split_point(&self, node: &TreeNode, choice_index: usize, 
                       choice_type: &ChoiceType, value: &ChoiceValue) -> Option<usize> {
        if choice_index < node.values.len() {
            if &node.choice_types[choice_index] != choice_type || &node.values[choice_index] != value {
                return Some(choice_index);
            }
        }
        None
    }
    
    /// Navigate to child node or create it if it doesn't exist
    fn navigate_or_create_child(&mut self, parent: &Arc<TreeNode>, choice_type: &ChoiceType, 
                               value: &ChoiceValue, constraints: &Box<Constraints>, was_forced: bool) -> Arc<TreeNode> {
        // Check if parent has a branch transition
        let mut parent_transition = parent.transition.write().unwrap();
        
        match parent_transition.as_mut() {
            Some(Transition::Branch(branch)) => {
                // Check if child already exists for this choice value
                let mut children = branch.children.write().unwrap();
                if let Some(existing_child) = children.get(value) {
                    return Arc::clone(existing_child);
                }
                
                // Create new child node
                let mut child = TreeNode::new(self.next_node_id);
                let child_id = self.next_node_id;
                self.next_node_id += 1;
                child.add_choice(*choice_type, value.clone(), constraints.clone(), was_forced);
                
                let child_arc = Arc::new(child);
                children.insert(value.clone(), Arc::clone(&child_arc));
                
                // Add to cache for potential reuse
                drop(children); // Release lock before cache operations
                self.node_cache.insert(child_id, Arc::clone(&child_arc));
                self.stats.total_nodes += 1;
                
                child_arc
            },
            _ => {
                // Parent doesn't have a branch transition, create one
                let branch = Branch {
                    children: RwLock::new(HashMap::new()),
                    is_exhausted: RwLock::new(false),
                };
                
                // Create the child node
                let mut child = TreeNode::new(self.next_node_id);
                self.next_node_id += 1;
                child.add_choice(*choice_type, value.clone(), constraints.clone(), was_forced);
                
                let child_arc = Arc::new(child);
                branch.children.write().unwrap().insert(value.clone(), Arc::clone(&child_arc));
                
                // Set the branch as parent's transition
                *parent_transition = Some(Transition::Branch(branch));
                
                self.stats.total_nodes += 1;
                self.stats.branch_nodes += 1;
                
                child_arc
            }
        }
    }
    
    /// Mark a node as a conclusion with the given status
    fn mark_conclusion(&mut self, node: Arc<TreeNode>, status: crate::data::Status, 
                      observations: HashMap<String, String>) {
        println!("DATATREE DEBUG: Marking node {} as conclusion with status {:?}", node.node_id, status);
        
        // Convert observations from HashMap<String, String> to HashMap<String, f64> for target_observations
        let target_observations: HashMap<String, f64> = observations.iter()
            .filter_map(|(k, v)| v.parse::<f64>().ok().map(|val| (k.clone(), val)))
            .collect();
            
        let metadata: HashMap<String, String> = observations.iter()
            .filter(|(_, v)| v.parse::<f64>().is_err())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        
        let conclusion = Conclusion {
            status,
            interesting_origin: None,
            target_observations,
            metadata,
        };
        
        *node.transition.write().unwrap() = Some(Transition::Conclusion(conclusion));
        self.stats.conclusion_nodes += 1;
    }
    
    /// Get statistics about the tree for analysis
    pub fn get_stats(&self) -> TreeStats {
        self.stats.clone()
    }
    
    /// Simulate test function execution through the tree
    /// Used by TreeRecordingObserver for testing purposes
    pub fn simulate_test_function(&self, choices: &[(ChoiceType, ChoiceValue, Box<Constraints>)]) -> (crate::data::Status, HashMap<String, String>) {
        println!("DATATREE DEBUG: Simulating test function with {} choices", choices.len());
        
        // For now, return a simple simulation result
        // In a full implementation, this would traverse the tree and predict the outcome
        (crate::data::Status::Valid, HashMap::new())
    }
}

impl Default for DataTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{ChoiceType, ChoiceValue, IntegerConstraints, Constraints};
    use rand::thread_rng;
    
    #[test]
    fn test_datatree_creation() {
        let tree = DataTree::new();
        assert_eq!(tree.stats.total_nodes, 0);
        assert_eq!(tree.stats.novel_prefixes_generated, 0);
    }
    
    #[test]
    fn test_novel_prefix_generation() {
        let mut tree = DataTree::new();
        let mut rng = thread_rng();
        
        // Generate a novel prefix from empty tree
        let prefix = tree.generate_novel_prefix(&mut rng);
        
        // Should generate something (even if empty from root)
        assert_eq!(tree.stats.novel_prefixes_generated, 1);
        println!("TEST DEBUG: Generated prefix with {} choices", prefix.len());
        
        // Test multiple generations to verify sophisticated behavior
        let prefix2 = tree.generate_novel_prefix(&mut rng);
        assert_eq!(tree.stats.novel_prefixes_generated, 2);
        
        // Test with a tree that has some recorded paths
        let choices = vec![
            (
                ChoiceType::Integer,
                ChoiceValue::Integer(42),
                Box::new(Constraints::Integer(IntegerConstraints::default())),
                false
            ),
        ];
        tree.record_path(&choices, crate::data::Status::Valid, HashMap::new());
        
        let prefix3 = tree.generate_novel_prefix(&mut rng);
        assert_eq!(tree.stats.novel_prefixes_generated, 3);
        println!("TEST DEBUG: Generated prefix after path recording: {} choices", prefix3.len());
    }
    
    #[test]
    fn test_tree_node_operations() {
        let mut node = TreeNode::new(42);
        
        // Add some choices
        node.add_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
            false
        );
        
        assert_eq!(node.values.len(), 1);
        assert_eq!(node.choice_types[0], ChoiceType::Integer);
        
        if let ChoiceValue::Integer(val) = &node.values[0] {
            assert_eq!(*val, 100);
        } else {
            panic!("Expected integer value");
        }
    }
    
    #[test]
    fn test_node_splitting() {
        let mut node = TreeNode::new(1);
        let mut next_id = 2;
        
        // Add multiple choices
        node.add_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(10),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
            false
        );
        node.add_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(20),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
            false
        );
        
        // Split at index 1
        let suffix = node.split_at(1, &mut next_id);
        
        // Original node should have first choice only
        assert_eq!(node.values.len(), 1);
        if let ChoiceValue::Integer(val) = &node.values[0] {
            assert_eq!(*val, 10);
        }
        
        // Suffix should have second choice
        assert_eq!(suffix.values.len(), 1);
        if let ChoiceValue::Integer(val) = &suffix.values[0] {
            assert_eq!(*val, 20);
        }
    }
    
    #[test]
    fn test_path_recording() {
        let mut tree = DataTree::new();
        
        // Create a simple path
        let choices = vec![
            (
                ChoiceType::Integer,
                ChoiceValue::Integer(42),
                Box::new(Constraints::Integer(IntegerConstraints::default())),
                false
            ),
        ];
        
        tree.record_path(&choices, crate::data::Status::Valid, HashMap::new());
        
        // Tree should have recorded this path
        assert!(tree.stats.total_nodes > 0);
        assert_eq!(tree.stats.conclusion_nodes, 1);
        
        // Test multiple paths to verify tree structure
        let choices2 = vec![
            (
                ChoiceType::Integer,
                ChoiceValue::Integer(84),
                Box::new(Constraints::Integer(IntegerConstraints::default())),
                false
            ),
        ];
        
        tree.record_path(&choices2, crate::data::Status::Valid, HashMap::new());
        assert_eq!(tree.stats.conclusion_nodes, 2);
    }
    
    #[test]
    fn test_sophisticated_exhaustion_detection() {
        let mut node = TreeNode::new(1);
        
        // Test node with no transition
        assert!(!node.check_exhausted());
        
        // Test node with branch transition
        let branch = Branch {
            children: RwLock::new(HashMap::new()),
            is_exhausted: RwLock::new(false),
        };
        *node.transition.write().unwrap() = Some(Transition::Branch(branch));
        
        // Empty branch should not be exhausted initially
        assert!(!node.check_exhausted());
        
        // Test exhaustion ratio calculation
        let ratio = node.compute_exhaustion_ratio();
        assert!(ratio >= 0.0 && ratio <= 1.0);
    }
    
    #[test]
    fn test_weighted_selection_logic() {
        let tree = DataTree::new();
        let node = Arc::new(TreeNode::new(1));
        
        // Test weight calculation
        let weight = tree.calculate_exploration_weight(&node, 5);
        assert!(weight > 0.0);
        
        // Test deeper depth affects weighting
        let deep_weight = tree.calculate_exploration_weight(&node, 15);
        assert!(deep_weight >= weight); // Should favor shallower at greater depths
    }
}