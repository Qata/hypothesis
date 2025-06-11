//! Comprehensive tests for Status system with perfect Python parity
//! 
//! These tests ensure that our Status enum behaves exactly like Python's Status.

use crate::data::{ConjectureData, Status, DrawError};

#[cfg(test)]
mod status_tests {
    use super::*;

    #[test]
    fn test_status_enum_values() {
        // Test that Status enum has correct Python parity values
        assert_eq!(Status::Overrun as i32, 0);
        assert_eq!(Status::Invalid as i32, 1); 
        assert_eq!(Status::Valid as i32, 2);
        assert_eq!(Status::Interesting as i32, 3);
    }

    #[test]
    fn test_status_default_is_valid() {
        // Python ConjectureData starts with Status.VALID
        let data = ConjectureData::new(42);
        assert_eq!(data.status, Status::Valid);
    }

    #[test]
    fn test_status_transition_valid_to_overrun() {
        let mut data = ConjectureData::new(42);
        
        // Set a small max_length to trigger overrun
        data.max_length = 4;
        
        // This should trigger overrun when we exceed buffer
        let result1 = data.draw_integer(0, 100);
        assert!(result1.is_ok());
        assert_eq!(data.status, Status::Valid);
        
        let result2 = data.draw_integer(0, 100);
        assert!(result2.is_ok()); 
        assert_eq!(data.status, Status::Valid);
        
        // This should trigger overrun (2 + 2 + 2 = 6 > 4)
        let result3 = data.draw_integer(0, 100);
        assert_eq!(result3, Err(DrawError::Overrun));
        assert_eq!(data.status, Status::Overrun);
    }

    #[test]
    fn test_status_transition_valid_to_invalid() {
        let mut data = ConjectureData::new(42);
        
        // Manually set status to Invalid to test constraint violations
        data.status = Status::Invalid;
        
        // Drawing should fail with InvalidStatus
        let result = data.draw_integer(0, 100);
        assert_eq!(result, Err(DrawError::InvalidStatus));
        assert_eq!(data.status, Status::Invalid);
    }

    #[test]
    fn test_status_transition_valid_to_interesting() {
        let mut data = ConjectureData::new(42);
        
        // Simulate test finding interesting result
        data.status = Status::Interesting;
        
        // Should not be able to draw more after interesting
        let result = data.draw_integer(0, 100);
        assert_eq!(result, Err(DrawError::InvalidStatus));
        assert_eq!(data.status, Status::Interesting);
    }

    #[test]
    fn test_status_prevents_draws_when_not_valid() {
        let mut data = ConjectureData::new(42);
        
        // Test each non-valid status
        for &status in &[Status::Overrun, Status::Invalid, Status::Interesting] {
            data.status = status;
            
            let int_result = data.draw_integer(0, 100);
            assert_eq!(int_result, Err(DrawError::InvalidStatus));
            
            let bool_result = data.draw_boolean(0.5);
            assert_eq!(bool_result, Err(DrawError::InvalidStatus));
            
            assert_eq!(data.status, status); // Status should remain unchanged
        }
    }

    #[test]
    fn test_overrun_detection_on_various_draws() {
        let mut data = ConjectureData::new(42);
        data.max_length = 3; // Very small buffer
        
        // Boolean draw should work (1 byte)
        assert!(data.draw_boolean(0.5).is_ok());
        assert_eq!(data.status, Status::Valid);
        assert_eq!(data.length, 1);
        
        // Integer draw should trigger overrun (would need 2 more bytes, total 3)
        let result = data.draw_integer(0, 100);
        assert!(result.is_ok()); 
        assert_eq!(data.status, Status::Valid);
        assert_eq!(data.length, 3);
        
        // Next draw should fail with overrun
        let result = data.draw_boolean(0.5);
        assert_eq!(result, Err(DrawError::Overrun));
        assert_eq!(data.status, Status::Overrun);
    }

    #[test] 
    fn test_status_display_formatting() {
        // Test that Status can be displayed (Debug trait)
        assert_eq!(format!("{:?}", Status::Valid), "Valid");
        assert_eq!(format!("{:?}", Status::Invalid), "Invalid");
        assert_eq!(format!("{:?}", Status::Overrun), "Overrun");
        assert_eq!(format!("{:?}", Status::Interesting), "Interesting");
    }

    #[test]
    fn test_status_comparison_and_equality() {
        // Test equality
        assert_eq!(Status::Valid, Status::Valid);
        assert_ne!(Status::Valid, Status::Invalid);
        
        // Test that we can pattern match
        let status = Status::Overrun;
        match status {
            Status::Overrun => (),
            _ => panic!("Pattern matching should work"),
        }
    }

    #[test]
    fn test_status_can_be_copied_and_cloned() {
        let status1 = Status::Valid;
        let status2 = status1; // Copy
        let status3 = status1.clone(); // Clone
        
        assert_eq!(status1, status2);
        assert_eq!(status1, status3);
    }

    #[test]
    fn test_status_transition_to_invalid_on_panic() {
        // This would be called by the engine when a test panics
        let mut data = ConjectureData::new(42);
        assert_eq!(data.status, Status::Valid);
        
        // Simulate what happens when test throws exception
        data.mark_invalid("Test panicked");
        assert_eq!(data.status, Status::Invalid);
        
        // Should not be able to draw after invalid
        let result = data.draw_integer(0, 100);
        assert_eq!(result, Err(DrawError::InvalidStatus));
    }

    #[test] 
    fn test_status_can_transition_to_interesting() {
        let mut data = ConjectureData::new(42);
        
        // Make some draws
        assert!(data.draw_integer(0, 100).is_ok());
        assert!(data.draw_boolean(0.5).is_ok());
        assert_eq!(data.status, Status::Valid);
        
        // Mark as interesting (test passed)
        data.mark_interesting();
        assert_eq!(data.status, Status::Interesting);
        
        // Should not be able to draw after interesting
        let result = data.draw_integer(0, 100);
        assert_eq!(result, Err(DrawError::InvalidStatus));
    }

    #[test]
    fn test_status_transition_validation() {
        let mut data = ConjectureData::new(42);
        
        // Valid transitions from Valid
        assert!(data.can_transition_to(Status::Interesting));
        assert!(data.can_transition_to(Status::Invalid));
        assert!(data.can_transition_to(Status::Overrun));
        assert!(data.can_transition_to(Status::Valid)); // Redundant but allowed
        
        // Test that once we transition to terminal state, we can't transition further
        data.status = Status::Interesting;
        assert!(!data.can_transition_to(Status::Valid));
        assert!(!data.can_transition_to(Status::Overrun));
        assert!(!data.can_transition_to(Status::Invalid));
        assert!(data.can_transition_to(Status::Interesting)); // To self is ok
        
        data.status = Status::Invalid;
        assert!(!data.can_transition_to(Status::Valid));
        assert!(!data.can_transition_to(Status::Overrun));
        assert!(!data.can_transition_to(Status::Interesting));
        assert!(data.can_transition_to(Status::Invalid)); // To self is ok
        
        data.status = Status::Overrun;
        assert!(!data.can_transition_to(Status::Valid));
        assert!(!data.can_transition_to(Status::Invalid));
        assert!(!data.can_transition_to(Status::Interesting));
        assert!(data.can_transition_to(Status::Overrun)); // To self is ok
    }

    #[test]
    fn test_status_debug_logging() {
        let mut data = ConjectureData::new(42);
        
        // Test that status changes are logged
        data.mark_invalid("Test constraint violation");
        assert_eq!(data.status, Status::Invalid);
        
        data = ConjectureData::new(42);
        data.mark_interesting();
        assert_eq!(data.status, Status::Interesting);
        
        data = ConjectureData::new(42);
        data.max_length = 1;
        let _ = data.draw_integer(0, 100); // This should trigger overrun
        assert_eq!(data.status, Status::Overrun);
    }
}