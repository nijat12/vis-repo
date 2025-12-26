"""
VM Utilities for Google Cloud VM Management

Provides functionality for:
- VM shutdown (killswitch)
- VM metadata retrieval
- Logging and audit trail
"""

import os
import sys
import time
import logging
import subprocess
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_vm_metadata() -> Optional[Dict[str, Any]]:
    """
    Retrieve VM instance metadata from Google Cloud.
    
    Returns:
        Dictionary with VM metadata or None if not running on GCP VM
    """
    try:
        # Try to get instance metadata
        result = subprocess.run(
            ["curl", "-s", "-H", "Metadata-Flavor: Google",
             "http://metadata.google.internal/computeMetadata/v1/instance/?recursive=true"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            import json
            metadata = json.loads(result.stdout)
            logger.info(f"Running on GCP VM: {metadata.get('name', 'unknown')}")
            return metadata
        else:
            logger.info("Not running on a GCP VM (metadata service unavailable)")
            return None
            
    except Exception as e:
        logger.debug(f"Could not retrieve VM metadata: {e}")
        return None


def shutdown_vm(delay_seconds: int = 60, force: bool = False) -> None:
    """
    Shutdown the VM instance after a delay.
    
    Args:
        delay_seconds: Seconds to wait before shutdown (gives time to review logs)
        force: If True, skip confirmation and metadata checks
    
    Raises:
        RuntimeError: If not running on a GCP VM (unless force=True)
    """
    # Check if we're on a GCP VM
    if not force:
        metadata = get_vm_metadata()
        if metadata is None:
            raise RuntimeError(
                "Not running on a GCP VM. Killswitch aborted. "
                "Use force=True to override this check."
            )
        
        vm_name = metadata.get('name', 'unknown')
        zone = metadata.get('zone', 'unknown').split('/')[-1]
        
        logger.warning(f"🔴 KILLSWITCH ACTIVATED - VM will shutdown in {delay_seconds} seconds")
        logger.warning(f"   VM Name: {vm_name}")
        logger.warning(f"   Zone: {zone}")
    else:
        logger.warning(f"🔴 KILLSWITCH ACTIVATED (FORCED) - System will shutdown in {delay_seconds} seconds")
    
    # Countdown
    for remaining in range(delay_seconds, 0, -10):
        if remaining <= 10:
            logger.warning(f"⏱️  Shutting down in {remaining} seconds...")
            time.sleep(1)
        else:
            logger.info(f"⏱️  Shutting down in {remaining} seconds...")
            time.sleep(10)
    
    logger.critical("🔴 EXECUTING SHUTDOWN NOW")
    
    try:
        # Flush all logs before shutdown
        logging.shutdown()
        
        # Execute shutdown command
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to execute shutdown command: {e}")
        logger.error("   You may need to configure sudo permissions for shutdown")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during shutdown: {e}")
        raise


def test_shutdown_permissions() -> bool:
    """
    Test if the current user has permissions to shutdown the VM.
    
    Returns:
        True if shutdown permissions are available, False otherwise
    """
    try:
        # Test sudo access without actually shutting down
        result = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            timeout=5
        )
        
        if result.returncode == 0:
            logger.info("✅ Shutdown permissions verified")
            return True
        else:
            logger.warning("⚠️  No sudo permissions - killswitch may not work")
            logger.warning("   Run: sudo visudo and add: <username> ALL=(ALL) NOPASSWD: /sbin/shutdown")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️  Could not verify shutdown permissions: {e}")
        return False


if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    
    print("Testing VM utilities...")
    print("\n1. Getting VM metadata:")
    metadata = get_vm_metadata()
    if metadata:
        print(f"   VM Name: {metadata.get('name')}")
        print(f"   Zone: {metadata.get('zone', '').split('/')[-1]}")
    
    print("\n2. Testing shutdown permissions:")
    test_shutdown_permissions()
    
    print("\n3. Killswitch test (dry run - will not actually shutdown):")
    print("   To test actual shutdown, run: python vm_utils.py --test-shutdown")
    
    if "--test-shutdown" in sys.argv:
        print("\n⚠️  WARNING: This will shutdown the VM in 10 seconds!")
        response = input("Type 'yes' to continue: ")
        if response.lower() == "yes":
            shutdown_vm(delay_seconds=10, force=True)
