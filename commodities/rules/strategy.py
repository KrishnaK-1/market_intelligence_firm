"""Back-compat shim: the strategy layer was promoted to the shared desks/
package when the firm grew beyond commodities. Import from desks.strategy."""
from desks.strategy import *  # noqa: F401,F403
