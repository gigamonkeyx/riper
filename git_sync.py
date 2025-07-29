#!/usr/bin/env python3
"""
RIPER-Ω Git Sync Automation
Automates pull/push for remote updates with conflict resolution
Observer-directed bias-free commit handling
"""

import subprocess
import logging
import sys
import os
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GitSyncManager:
    """Automated git sync with conflict resolution"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.sync_results = {
            "pull_status": "pending",
            "push_status": "pending",
            "conflicts": [],
            "commit_hash": None,
            "files_changed": 0
        }
    
    def run_git_command(self, command: list) -> Dict[str, Any]:
        """Execute git command and return results"""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def get_current_commit(self) -> Optional[str]:
        """Get current commit hash"""
        result = self.run_git_command(["git", "rev-parse", "--short", "HEAD"])
        if result["success"]:
            return result["stdout"]
        return None
    
    def check_remote_changes(self) -> bool:
        """Check if remote has changes"""
        # Fetch remote changes
        fetch_result = self.run_git_command(["git", "fetch", "origin", "main"])
        if not fetch_result["success"]:
            logger.warning(f"Fetch failed: {fetch_result['stderr']}")
            return False

        # Check if remote is ahead using rev-list
        ahead_result = self.run_git_command(["git", "rev-list", "--count", "HEAD..origin/main"])
        if ahead_result["success"]:
            ahead_count = int(ahead_result["stdout"]) if ahead_result["stdout"].isdigit() else 0
            return ahead_count > 0

        # Fallback: check status
        status_result = self.run_git_command(["git", "status", "-uno"])
        if status_result["success"]:
            return "Your branch is behind" in status_result["stdout"] or "have diverged" in status_result["stdout"]
        return False
    
    def pull_changes(self) -> bool:
        """Pull remote changes with conflict handling"""
        logger.info("Pulling remote changes...")
        
        # Check for uncommitted changes
        status_result = self.run_git_command(["git", "status", "--porcelain"])
        if status_result["success"] and status_result["stdout"]:
            logger.warning("Uncommitted changes detected, stashing...")
            stash_result = self.run_git_command(["git", "stash", "push", "-m", "Auto-stash before sync"])
            if not stash_result["success"]:
                logger.error(f"Stash failed: {stash_result['stderr']}")
                self.sync_results["pull_status"] = "failed"
                return False
        
        # Pull changes
        pull_result = self.run_git_command(["git", "pull", "origin", "main"])
        
        if pull_result["success"]:
            logger.info("Pull successful")
            self.sync_results["pull_status"] = "success"
            return True
        else:
            logger.error(f"Pull failed: {pull_result['stderr']}")
            self.sync_results["pull_status"] = "failed"
            
            # Check for merge conflicts
            if "CONFLICT" in pull_result["stderr"] or "Automatic merge failed" in pull_result["stderr"]:
                self.sync_results["conflicts"].append("Merge conflict detected")
                logger.error("Merge conflicts detected - manual resolution required")
            
            return False
    
    def push_changes(self) -> bool:
        """Push local changes to remote"""
        logger.info("Pushing local changes...")
        
        # Get current commit for tracking
        current_commit = self.get_current_commit()
        if current_commit:
            self.sync_results["commit_hash"] = current_commit
        
        # Push changes
        push_result = self.run_git_command(["git", "push", "origin", "main"])
        
        if push_result["success"]:
            logger.info(f"Push successful - Commit: {current_commit}")
            self.sync_results["push_status"] = "success"
            return True
        else:
            logger.error(f"Push failed: {push_result['stderr']}")
            self.sync_results["push_status"] = "failed"
            return False
    
    def sync_repository(self) -> Dict[str, Any]:
        """Complete sync operation: pull then push"""
        logger.info("Starting git sync operation...")
        
        # Check if remote has changes
        has_remote_changes = self.check_remote_changes()
        
        if has_remote_changes:
            logger.info("Remote changes detected, pulling first...")
            if not self.pull_changes():
                logger.error("Pull failed, aborting sync")
                return self.sync_results
        else:
            logger.info("No remote changes detected")
            self.sync_results["pull_status"] = "not_needed"
        
        # Push local changes
        if not self.push_changes():
            logger.error("Push failed")
            return self.sync_results
        
        # Get final status
        final_commit = self.get_current_commit()
        if final_commit:
            self.sync_results["commit_hash"] = final_commit
        
        # Count changed files
        diff_result = self.run_git_command(["git", "diff", "--name-only", "HEAD~1", "HEAD"])
        if diff_result["success"]:
            self.sync_results["files_changed"] = len(diff_result["stdout"].split('\n')) if diff_result["stdout"] else 0
        
        logger.info(f"Git sync complete - Commit: {final_commit}, Files: {self.sync_results['files_changed']}")
        return self.sync_results

def main():
    """Main sync execution"""
    sync_manager = GitSyncManager()
    
    try:
        results = sync_manager.sync_repository()
        
        # Log results factually
        if results["pull_status"] == "success" and results["push_status"] == "success":
            logger.info(f"Git sync: Success. Commit: {results['commit_hash']}")
            print(f"SUCCESS: Git sync completed - Commit {results['commit_hash']}")
            return 0
        elif results["push_status"] == "success" and results["pull_status"] == "not_needed":
            logger.info(f"Git sync: Success. Commit: {results['commit_hash']}")
            print(f"SUCCESS: Git push completed - Commit {results['commit_hash']}")
            return 0
        else:
            logger.error(f"Git sync: Failure. Pull: {results['pull_status']}, Push: {results['push_status']}")
            print(f"FAILURE: Git sync failed - Pull: {results['pull_status']}, Push: {results['push_status']}")
            if results["conflicts"]:
                print(f"Conflicts: {', '.join(results['conflicts'])}")
            return 1
            
    except Exception as e:
        logger.error(f"Git sync error: {e}")
        print(f"ERROR: Git sync failed - {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
