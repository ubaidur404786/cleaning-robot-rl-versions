"""
================================================================================
UTILITY HELPERS - Common Utility Functions
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning (Q-Learning)
FILE: utils/helpers.py
PURPOSE: Reusable helper functions for training and testing

================================================================================
📚 OVERVIEW
================================================================================

This module provides common utility functions used across the project:
- Time formatting for training duration display
- Progress bar for visual training progress
- Console output formatting helpers

================================================================================
"""

import sys
import time


def format_time(seconds):
    """
    Format a duration in seconds to a human-readable string.
    
    Converts seconds to HH:MM:SS format for easy reading.
    Useful for displaying training duration and estimated time.
    
    Parameters:
    -----------
    seconds : float
        Duration in seconds to format
    
    Returns:
    --------
    str
        Formatted time string (e.g., "01:23:45" or "00:05:30")
    
    Examples:
    ---------
    >>> format_time(90)
    '00:01:30'
    >>> format_time(3661)
    '01:01:01'
    >>> format_time(45.7)
    '00:00:45'
    """
    # Convert to integer seconds
    total_seconds = int(seconds)
    
    # Calculate hours, minutes, seconds
    hours = total_seconds // 3600           # 1 hour = 3600 seconds
    minutes = (total_seconds % 3600) // 60  # Remaining minutes
    secs = total_seconds % 60               # Remaining seconds
    
    # Format as HH:MM:SS
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds):
    """
    Format duration with appropriate units.
    
    Chooses the most appropriate unit based on the duration:
    - Under 1 minute: shows seconds
    - Under 1 hour: shows minutes and seconds
    - 1 hour or more: shows hours, minutes, seconds
    
    Parameters:
    -----------
    seconds : float
        Duration in seconds
    
    Returns:
    --------
    str
        Human-readable duration string
    
    Examples:
    ---------
    >>> format_duration(45)
    '45s'
    >>> format_duration(125)
    '2m 5s'
    >>> format_duration(3725)
    '1h 2m 5s'
    """
    total_seconds = int(seconds)
    
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        secs = total_seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours}h {minutes}m {secs}s"


def print_progress_bar(
    current,
    total,
    prefix="Progress",
    suffix="",
    length=40,
    fill="█",
    empty="░"
):
    """
    Print a progress bar to console.
    
    Creates a visual progress indicator that updates in place,
    useful for showing training or testing progress.
    
    Parameters:
    -----------
    current : int
        Current progress value (e.g., current episode)
    total : int
        Total value to reach (e.g., total episodes)
    prefix : str
        Text to show before the progress bar
    suffix : str
        Text to show after the progress bar
    length : int
        Width of the progress bar in characters
    fill : str
        Character to use for completed portion
    empty : str
        Character to use for incomplete portion
    
    Example Output:
    ---------------
    Progress |████████████░░░░░░░░░░░░░░░░░░| 40.0% (400/1000) ETA: 00:02:30
    """
    # Calculate percentage
    percent = current / total * 100 if total > 0 else 0
    
    # Calculate filled length
    filled_length = int(length * current // total) if total > 0 else 0
    
    # Create bar string
    bar = fill * filled_length + empty * (length - filled_length)
    
    # Print progress bar (use \r to overwrite previous line)
    progress_str = f"\r  {prefix} |{bar}| {percent:5.1f}% ({current}/{total}) {suffix}"
    sys.stdout.write(progress_str)
    sys.stdout.flush()
    
    # Print newline when complete
    if current >= total:
        print()


def print_header(title, width=70, char="="):
    """
    Print a formatted header with decorative borders.
    
    Creates consistent visual headers throughout the application.
    
    Parameters:
    -----------
    title : str
        Title text to display
    width : int
        Total width of the header
    char : str
        Character to use for borders
    
    Example Output:
    ---------------
    ======================================================================
      TRAINING Q-LEARNING AGENT
    ======================================================================
    """
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def print_divider(width=70, char="-"):
    """
    Print a simple divider line.
    
    Parameters:
    -----------
    width : int
        Width of the divider
    char : str
        Character to use
    """
    print(char * width)


def print_key_value(key, value, key_width=20):
    """
    Print a key-value pair with consistent formatting.
    
    Parameters:
    -----------
    key : str
        Label/key text
    value : any
        Value to display
    key_width : int
        Width to allocate for key (for alignment)
    
    Example Output:
    ---------------
        Episodes:           3000
        Learning Rate:      0.2
    """
    print(f"    {key:<{key_width}} {value}")


def create_running_average(data, window=100):
    """
    Calculate running average of a data series.
    
    Useful for smoothing noisy training metrics to see trends.
    
    Parameters:
    -----------
    data : list
        List of values to average
    window : int
        Size of the moving average window
    
    Returns:
    --------
    list
        Running average values
    
    Example:
    --------
    >>> rewards = [10, 20, 30, 40, 50]
    >>> create_running_average(rewards, window=3)
    [10, 15, 20, 30, 40]  # Average of last 3 values at each point
    """
    running_avg = []
    
    for i in range(len(data)):
        # Get window of values (or all values if less than window)
        start_idx = max(0, i - window + 1)
        window_data = data[start_idx:i + 1]
        
        # Calculate average
        avg = sum(window_data) / len(window_data)
        running_avg.append(avg)
    
    return running_avg


def clamp(value, min_value, max_value):
    """
    Clamp a value to a specified range.
    
    Ensures value stays within bounds.
    
    Parameters:
    -----------
    value : float
        Value to clamp
    min_value : float
        Minimum allowed value
    max_value : float
        Maximum allowed value
    
    Returns:
    --------
    float
        Clamped value
    
    Example:
    --------
    >>> clamp(150, 0, 100)
    100
    >>> clamp(-10, 0, 100)
    0
    """
    return max(min_value, min(max_value, value))


class Timer:
    """
    Simple timer class for measuring elapsed time.
    
    Useful for timing training duration, episode times, etc.
    
    Usage:
    ------
    timer = Timer()
    timer.start()
    # ... do something ...
    elapsed = timer.elapsed()
    print(f"Took {elapsed:.2f} seconds")
    """
    
    def __init__(self):
        """Initialize the timer."""
        self._start_time = None
        self._end_time = None
    
    def start(self):
        """Start the timer."""
        self._start_time = time.time()
        self._end_time = None
        return self
    
    def stop(self):
        """Stop the timer."""
        self._end_time = time.time()
        return self.elapsed()
    
    def elapsed(self):
        """
        Get elapsed time in seconds.
        
        Returns:
        --------
        float
            Elapsed time in seconds
        """
        if self._start_time is None:
            return 0.0
        
        end = self._end_time if self._end_time else time.time()
        return end - self._start_time
    
    def elapsed_formatted(self):
        """
        Get elapsed time as formatted string.
        
        Returns:
        --------
        str
            Formatted elapsed time
        """
        return format_time(self.elapsed())


# ================================================================================
# MODULE TEST
# ================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TESTING HELPER FUNCTIONS")
    print("=" * 60)
    
    # Test format_time
    print("\n1. Testing format_time():")
    test_times = [45, 90, 3600, 3661, 7325]
    for t in test_times:
        print(f"   {t:5d} seconds = {format_time(t)}")
    
    # Test format_duration
    print("\n2. Testing format_duration():")
    for t in test_times:
        print(f"   {t:5d} seconds = {format_duration(t)}")
    
    # Test progress bar
    print("\n3. Testing print_progress_bar():")
    for i in range(0, 101, 20):
        print_progress_bar(i, 100, prefix="Loading")
        time.sleep(0.2)
    
    # Test running average
    print("\n4. Testing create_running_average():")
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    avg = create_running_average(data, window=3)
    print(f"   Data: {data}")
    print(f"   Avg:  {[round(x, 1) for x in avg]}")
    
    # Test Timer
    print("\n5. Testing Timer class:")
    timer = Timer()
    timer.start()
    time.sleep(0.5)
    print(f"   Elapsed: {timer.elapsed():.3f}s")
    time.sleep(0.5)
    print(f"   Elapsed: {timer.elapsed_formatted()}")
    
    # Test headers
    print("\n6. Testing print_header():")
    print_header("EXAMPLE HEADER", width=60)
    
    print("\nAll tests passed!")
