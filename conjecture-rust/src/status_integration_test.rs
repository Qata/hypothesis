//! Integration test demonstrating the complete Status system implementation
//! 
//! This test verifies that the Status system works correctly across all components
//! and maintains perfect Python parity in behavior and values.

use crate::data::{ConjectureData, Status};
use std::collections::HashMap;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_status_system_workflow() {
        // Test 1: Normal test execution flow
        let mut data = ConjectureData::new(42);
        
        // Initially VALID
        assert_eq!(data.status, Status::Valid);
        assert_eq!(Status::Valid as i32, 2); // Python parity
        
        // Can make draws while valid
        assert!(data.draw_integer(0, 100).is_ok());
        assert!(data.draw_boolean(0.5).is_ok());
        assert_eq!(data.status, Status::Valid);
        
        // Test 2: Transition to INTERESTING
        data.mark_interesting();
        assert_eq!(data.status, Status::Interesting);
        assert_eq!(Status::Interesting as i32, 3); // Python parity
        
        // Cannot draw after interesting
        assert!(data.draw_integer(0, 100).is_err());
        
        // Test 3: OVERRUN detection
        let mut data2 = ConjectureData::new(123);
        data2.max_length = 2; // Very small buffer
        
        assert!(data2.draw_boolean(0.5).is_ok()); // 1 byte
        assert!(data2.draw_boolean(0.5).is_ok()); // 2 bytes total 
        
        // Next draw should trigger overrun
        assert!(data2.draw_integer(0, 100).is_err()); // Would need 2 more bytes
        assert_eq!(data2.status, Status::Overrun);
        assert_eq!(Status::Overrun as i32, 0); // Python parity
        
        // Test 4: INVALID status handling
        let mut data3 = ConjectureData::new(456);
        data3.mark_invalid("Test constraint violation");
        assert_eq!(data3.status, Status::Invalid);
        assert_eq!(Status::Invalid as i32, 1); // Python parity
        
        // Should store reason in events
        assert!(data3.events.contains_key("invalid_reason"));
        assert_eq!(data3.events["invalid_reason"], "Test constraint violation");
        
        // Cannot draw after invalid
        assert!(data3.draw_integer(0, 100).is_err());
    }
    
    #[test]
    fn test_status_transition_rules() {
        let mut data = ConjectureData::new(789);
        
        // From VALID, can transition anywhere
        assert!(data.can_transition_to(Status::Valid));
        assert!(data.can_transition_to(Status::Invalid));
        assert!(data.can_transition_to(Status::Interesting));
        assert!(data.can_transition_to(Status::Overrun));
        
        // Transition to terminal state
        data.status = Status::Interesting;
        
        // From terminal state, can only transition to self
        assert!(data.can_transition_to(Status::Interesting));
        assert!(!data.can_transition_to(Status::Valid));
        assert!(!data.can_transition_to(Status::Invalid));
        assert!(!data.can_transition_to(Status::Overrun));
        
        // Test other terminal states
        data.status = Status::Invalid;
        assert!(data.can_transition_to(Status::Invalid));
        assert!(!data.can_transition_to(Status::Valid));
        
        data.status = Status::Overrun;
        assert!(data.can_transition_to(Status::Overrun));
        assert!(!data.can_transition_to(Status::Valid));
    }
    
    #[test]
    fn test_status_with_datatree_integration() {
        use crate::data::TreeRecordingObserver;
        
        let mut data = ConjectureData::new(999);
        let mut observer = TreeRecordingObserver::new();
        
        // Record some choices
        assert!(data.draw_integer(1, 10).is_ok());
        assert!(data.draw_boolean(0.7).is_ok());
        
        // Finalize with different statuses and ensure it works
        observer.finalize_path(Status::Valid, HashMap::new());
        observer.finalize_path(Status::Interesting, HashMap::new());
        observer.finalize_path(Status::Invalid, HashMap::new());
        observer.finalize_path(Status::Overrun, HashMap::new());
        
        // Should not panic or fail
        assert_eq!(data.status, Status::Valid);
    }
    
    #[test]
    fn test_status_enum_completeness() {
        // Verify all Python Status values are present and correct
        let statuses = [
            (Status::Overrun, 0),
            (Status::Invalid, 1),
            (Status::Valid, 2),
            (Status::Interesting, 3),
        ];
        
        for (status, expected_value) in statuses {
            assert_eq!(status as i32, expected_value, 
                      "Status {:?} should have value {}", status, expected_value);
        }
    }
    
    #[test]
    fn test_status_debug_output() {
        let mut data = ConjectureData::new(111);
        
        // Test that status changes produce debug output
        // (This just ensures the methods can be called without panicking)
        data.mark_invalid("Test debug output");
        data = ConjectureData::new(222);
        data.mark_interesting();
        data = ConjectureData::new(333);
        data.max_length = 1;
        let _ = data.draw_integer(0, 100); // Should trigger overrun
        
        // All debug calls should complete without panicking
    }
    
    #[test]
    fn test_terminal_status_detection() {
        let mut data = ConjectureData::new(444);
        
        // VALID is not terminal
        assert!(!data.is_terminal());
        assert!(data.can_draw());
        
        // INTERESTING is terminal
        data.status = Status::Interesting;
        assert!(data.is_terminal());
        assert!(!data.can_draw());
        
        // INVALID is terminal
        data.status = Status::Invalid;
        assert!(data.is_terminal());
        assert!(!data.can_draw());
        
        // OVERRUN is terminal
        data.status = Status::Overrun;
        assert!(data.is_terminal());
        assert!(!data.can_draw());
    }
}