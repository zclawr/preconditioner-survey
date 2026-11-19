def index_to_grid(r,n):
    """Convert a row number into a grid point."""
    return (r // n, r % n)