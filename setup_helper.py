"""
VERIFAI Setup and Migration Utility

Quick commands for:
1. Testing database connections
2. Migrating from SQLite to Supabase
3. Validating doctor feedback setup
4. Running health checks

Usage:
    python setup_helper.py check-db
    python setup_helper.py migrate
    python setup_helper.py test-feedback
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_database():
    """Check database connection health."""
    print("\n" + "="*60)
    print("DATABASE HEALTH CHECK")
    print("="*60 + "\n")
    
    from db.adapter import check_database_health
    from app.config import settings
    
    health = check_database_health()
    
    print(f"Mode: {health['mode'].upper()}")
    print(f"Status: {'✅ HEALTHY' if health['healthy'] else '❌ FAILED'}")
    print(f"\nDetails:")
    for key, value in health['details'].items():
        print(f"  {key}: {value}")
    
    if health['mode'] == 'supabase':
        print(f"\nSupabase Configuration:")
        print(f"  URL: {settings.SUPABASE_URL}")
        print(f"  Key: {settings.SUPABASE_KEY[:20]}..." if settings.SUPABASE_KEY else "  Key: NOT SET")
    else:
        print(f"\nSQLite Configuration:")
        from db.connection import DB_PATH
        print(f"  Path: {DB_PATH}")
    
    print("\n" + "="*60 + "\n")
    return health['healthy']


def migrate_database():
    """Migrate from SQLite to Supabase."""
    print("\n" + "="*60)
    print("DATABASE MIGRATION: SQLite → Supabase")
    print("="*60 + "\n")
    
    from db.connection import DB_PATH
    from db.adapter import migrate_to_cloud
    from app.config import settings
    
    # Validate prerequisites
    if not os.path.exists(DB_PATH):
        print(f"❌ ERROR: SQLite database not found at {DB_PATH}")
        print("Nothing to migrate. Run VERIFAI first to create data.")
        return False
    
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        print("❌ ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        return False
    
    print(f"Source: {DB_PATH}")
    print(f"Target: {settings.SUPABASE_URL}")
    print(f"\n⚠️  WARNING: This will copy all data from SQLite to Supabase.")
    print("Make sure you have:")
    print("  1. Created Supabase project")
    print("  2. Run db/supabase_schema.sql in SQL Editor")
    print("  3. Set correct SUPABASE_URL and SUPABASE_KEY in .env")
    
    confirm = input("\nContinue? (type 'yes' to proceed): ")
    if confirm.lower() != 'yes':
        print("Migration cancelled.")
        return False
    
    try:
        migrate_to_cloud(DB_PATH)
        print("\n✅ Migration completed successfully!")
        print("\nNext steps:")
        print("  1. Set DATABASE_MODE=supabase in .env")
        print("  2. Restart your application")
        print("  3. Test connection with: python setup_helper.py check-db")
        return True
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return False


def test_feedback_flow():
    """Test doctor feedback functionality."""
    print("\n" + "="*60)
    print("DOCTOR FEEDBACK TEST")
    print("="*60 + "\n")
    
    from agents.feedback import (
        capture_doctor_feedback,
        prepare_feedback_for_reprocessing,
        create_feedback_enhanced_state
    )
    from db.adapter import get_logger
    
    # Check if we have any sessions
    sessions = get_logger.list_sessions(limit=1)
    
    if not sessions:
        print("❌ No workflow sessions found.")
        print("Run a diagnosis workflow first before testing feedback.")
        return False
    
    latest_session = sessions[0]
    session_id = latest_session['session_id']
    
    print(f"Testing with latest session: {session_id}")
    print(f"Original diagnosis: {latest_session.get('final_diagnosis', 'N/A')}")
    print(f"Original confidence: {latest_session.get('final_confidence', 0):.2%}")
    
    print("\n1. Capturing test feedback...")
    try:
        feedback_id = capture_doctor_feedback(
            session_id=session_id,
            feedback_type="rejection",
            doctor_notes="Test feedback - this is a simulation to verify the feedback system works.",
            correct_diagnosis="Test corrected diagnosis",
            rejection_reasons=["test_reason"],
            doctor_id="test_doctor"
        )
        print(f"✅ Feedback captured: ID = {feedback_id}")
    except Exception as e:
        print(f"❌ Failed to capture feedback: {e}")
        return False
    
    print("\n2. Preparing feedback for reprocessing...")
    try:
        feedback_input = prepare_feedback_for_reprocessing(feedback_id)
        print(f"✅ Feedback prepared")
        print(f"   Original session: {feedback_input.original_session_id}")
        print(f"   Doctor notes: {feedback_input.doctor_notes[:50]}...")
    except Exception as e:
        print(f"❌ Failed to prepare feedback: {e}")
        return False
    
    print("\n3. Creating enhanced state...")
    try:
        new_state = create_feedback_enhanced_state(
            feedback_input=feedback_input,
            image_path=latest_session.get('image_path', 'test.jpg'),
            patient_id=latest_session.get('patient_id')
        )
        print(f"✅ State created")
        print(f"   New session ID: {new_state['_session_id']}")
        print(f"   Is feedback iteration: {new_state['is_feedback_iteration']}")
        print(f"   Has doctor feedback: {new_state['doctor_feedback'] is not None}")
    except Exception as e:
        print(f"❌ Failed to create state: {e}")
        return False
    
    print("\n✅ Doctor feedback system is working correctly!")
    print("\nNote: This was a test. To actually reprocess:")
    print("  from graph.workflow import app")
    print("  result = app.invoke(new_state)")
    
    return True


def show_stats():
    """Show database statistics."""
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60 + "\n")
    
    from db.adapter import get_logger
    
    try:
        stats = get_logger.get_diagnosis_stats()
        
        print(f"Total Sessions: {stats['total_sessions']}")
        print(f"Completed: {stats['completed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Deferred: {stats['deferred']}")
        print(f"\nAverage Confidence: {stats['avg_confidence']:.2%}" if stats['avg_confidence'] else "N/A")
        print(f"Avg Agents per Session: {stats['avg_agents_per_session']:.1f}" if stats['avg_agents_per_session'] else "N/A")
        print(f"Debate Consensus Rate: {stats['debate_consensus_rate']:.1%}" if stats['debate_consensus_rate'] else "N/A")
        
        if stats['top_diagnoses']:
            print(f"\nTop Diagnoses:")
            for dx in stats['top_diagnoses'][:5]:
                print(f"  • {dx['final_diagnosis']}: {dx['cnt']} times (avg conf: {dx['avg_conf']:.1%})")
    except Exception as e:
        print(f"❌ Failed to retrieve stats: {e}")
        return False
    
    return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("\nVERIFAI Setup and Migration Utility")
        print("="*60)
        print("\nCommands:")
        print("  check-db        - Check database connection health")
        print("  migrate         - Migrate from SQLite to Supabase")
        print("  test-feedback   - Test doctor feedback functionality")
        print("  stats           - Show database statistics")
        print("\nUsage:")
        print("  python setup_helper.py <command>")
        print("\nExample:")
        print("  python setup_helper.py check-db")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == 'check-db':
        success = check_database()
        sys.exit(0 if success else 1)
    
    elif command == 'migrate':
        success = migrate_database()
        sys.exit(0 if success else 1)
    
    elif command == 'test-feedback':
        success = test_feedback_flow()
        sys.exit(0 if success else 1)
    
    elif command == 'stats':
        success = show_stats()
        sys.exit(0 if success else 1)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python setup_helper.py' to see available commands")
        sys.exit(1)


if __name__ == "__main__":
    main()
