"""Isolated XR0 subclass; same architecture, parameter keys and inference path."""
from mibot.models.VLA.XR0 import XR0
from contact_loss import ContactLossMixin


class ContactWeightedXR0(ContactLossMixin, XR0):
    pass
