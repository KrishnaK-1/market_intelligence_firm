"""Back-compat shim: the signal library was promoted to the shared desks/
package when the firm grew beyond commodities. Import from desks.signals."""
from desks.signals import *  # noqa: F401,F403
