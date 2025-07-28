"""
Commit Verification Tool for RIPER-Ω System
Logs new commit hashes and tracks file changes for audit trail.
"""

import subprocess
import logging
import json
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class CommitChecker:
    """Tool for verifying and logging git commits"""
    
    def __init__(self):
        self.commit_history = []
    
    def get_latest_commit_info(self) -> Dict[str, Any]:
        """Get information about the latest commit"""
        try:
            # Get latest commit hash
            hash_result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            commit_hash = hash_result.stdout.strip()
            
            # Get commit message
            message_result = subprocess.run(
                ['git', 'log', '-1', '--pretty=format:%s'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            commit_message = message_result.stdout.strip()
            
            # Get changed files
            files_result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            changed_files = files_result.stdout.strip().split('\n') if files_result.stdout.strip() else []
            
            # Get commit stats
            stats_result = subprocess.run(
                ['git', 'show', '--stat', '--format=', 'HEAD'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse insertions and deletions
            stats_lines = stats_result.stdout.strip().split('\n')
            insertions = 0
            deletions = 0
            files_changed = len(changed_files)
            
            for line in stats_lines:
                if 'insertion' in line or 'deletion' in line:
                    parts = line.split(',')
                    for part in parts:
                        if 'insertion' in part:
                            insertions = int(part.strip().split()[0])
                        elif 'deletion' in part:
                            deletions = int(part.strip().split()[0])
            
            commit_info = {
                "hash": commit_hash,
                "short_hash": commit_hash[:8],
                "message": commit_message,
                "changed_files": changed_files,
                "files_changed": files_changed,
                "insertions": insertions,
                "deletions": deletions,
                "timestamp": datetime.now().isoformat()
            }
            
            return commit_info
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return {
                "hash": "unknown",
                "short_hash": "unknown",
                "message": "error",
                "changed_files": [],
                "files_changed": 0,
                "insertions": 0,
                "deletions": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def check_push_status(self) -> Dict[str, Any]:
        """Check if latest commit has been pushed to remote"""
        try:
            # Check if local is ahead of remote
            status_result = subprocess.run(
                ['git', 'status', '--porcelain=v1', '--branch'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            status_lines = status_result.stdout.strip().split('\n')
            branch_line = status_lines[0] if status_lines else ""
            
            if '[ahead' in branch_line:
                push_status = "pending"
                commits_ahead = int(branch_line.split('[ahead ')[1].split(']')[0])
            else:
                push_status = "pushed"
                commits_ahead = 0
            
            return {
                "status": push_status,
                "commits_ahead": commits_ahead,
                "branch_info": branch_line
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git status check failed: {e}")
            return {
                "status": "error",
                "commits_ahead": 0,
                "branch_info": "",
                "error": str(e)
            }
    
    def log_commit_verification(self) -> Dict[str, Any]:
        """Log commit verification with detailed information"""
        commit_info = self.get_latest_commit_info()
        push_info = self.check_push_status()
        
        # Combine information
        verification_result = {
            **commit_info,
            "push_status": push_info["status"],
            "commits_ahead": push_info["commits_ahead"]
        }
        
        # Add to history
        self.commit_history.append(verification_result)
        
        # Log factually
        logger.info(f"Commit: Hash {commit_info['short_hash']}. "
                   f"Files: {commit_info['files_changed']} changed "
                   f"(+{commit_info['insertions']}/-{commit_info['deletions']}). "
                   f"Status: {push_info['status'].title()}")
        
        return verification_result
    
    def save_commit_log(self, filename: str = "commit_verification.json"):
        """Save commit verification history to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.commit_history, f, indent=2)
            logger.info(f"Commit log saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save commit log: {e}")

def main():
    """Main function for standalone usage"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    checker = CommitChecker()
    result = checker.log_commit_verification()
    
    print(f"Latest Commit: {result['short_hash']}")
    print(f"Message: {result['message']}")
    print(f"Files Changed: {result['files_changed']}")
    print(f"Push Status: {result['push_status']}")
    
    return result

if __name__ == "__main__":
    main()
