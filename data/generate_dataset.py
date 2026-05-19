import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

random.seed(42)
np.random.seed(42)


@dataclass
class IntentTemplate:
    intent: str
    templates: List[str]
    priority: str
    department: str


INTENT_TEMPLATES = [
    IntentTemplate(
        intent="payment_issue",
        priority="critical",
        department="Billing Team",
        templates=[
            "My payment failed but money got deducted",
            "I was charged twice for the same transaction",
            "Payment is showing failed but amount was debited from my account",
            "Transaction declined but I can see the charge on my bank statement",
            "Double charged for my subscription this month",
            "My card was charged but order shows payment failed",
            "Payment error occurred and money deducted from my wallet",
            "Unable to complete payment keep getting error",
            "Charged but transaction shows unsuccessful",
            "My UPI payment failed but balance was deducted",
            "Bank shows debit but your system shows payment pending",
            "Got charged for a failed transaction need immediate help",
            "Payment gateway showing error but money left my account",
            "Duplicate payment processed on my credit card",
            "Net banking shows debit but order not confirmed",
        ],
    ),
    IntentTemplate(
        intent="refund_request",
        priority="high",
        department="Billing Team",
        templates=[
            "I want to request a refund for my last order",
            "Please refund the amount for my recent order",
            "I need my money back product was not as described",
            "Requesting refund due to delayed delivery",
            "How do I get a refund for a cancelled order",
            "My refund has not been processed after many days",
            "Need refund for defective product received",
            "Please initiate refund for duplicate charge",
            "When will my refund be credited back to my account",
            "Refund status not updated",
            "I cancelled the order but have not received the refund yet",
            "Refund amount is incorrect",
            "Requesting full refund item was damaged on arrival",
            "Partial refund received rest of the amount is pending",
            "No refund yet since cancellation",
        ],
    ),
    IntentTemplate(
        intent="account_locked",
        priority="critical",
        department="Security Team",
        templates=[
            "My account has been locked and I cannot login",
            "Account is disabled I did not violate any terms",
            "Cannot access my account showing suspended message",
            "My account got locked after multiple failed attempts",
            "Account suspended without any notification or email",
            "Locked out of my account need immediate access",
            "Two factor authentication not working account locked",
            "Someone may have hacked my account it is now locked",
            "Account blocked I have been a customer for years",
            "Cannot login account shows restricted status",
            "Email says account suspended but I do not know why",
            "Phone number changed now locked out of account",
            "Account verification failed multiple times now locked",
            "Unable to login after password reset account locked",
            "My account was locked due to suspicious activity alert",
        ],
    ),
    IntentTemplate(
        intent="technical_bug",
        priority="high",
        department="Engineering Team",
        templates=[
            "App keeps crashing when I try to open my dashboard",
            "Website not loading properly on Chrome browser",
            "Getting error 500 when trying to submit the form",
            "The mobile app freezes on the checkout screen",
            "Search functionality is broken showing wrong results",
            "Images not loading on the product page",
            "API returning 404 for endpoint that was working yesterday",
            "Dark mode not saving after I close the app",
            "Notification settings keep resetting to default",
            "The export to CSV feature is generating empty files",
            "Login button not responding on mobile devices",
            "Getting blank screen after updating to latest version",
            "Charts not displaying correctly in the analytics section",
            "File upload failing with no error message shown",
            "Browser console showing JavaScript error on home page",
        ],
    ),
    IntentTemplate(
        intent="feature_request",
        priority="low",
        department="Product Team",
        templates=[
            "Can you add dark mode to the mobile app",
            "It would be great to have bulk export functionality",
            "Please add support for multiple currencies",
            "Would love to see calendar integration in the dashboard",
            "Can you add two-factor authentication via SMS",
            "Feature request allow custom report templates",
            "Please add keyboard shortcuts for power users",
            "Can we have an API for integrating with third party tools",
            "Would be useful to have a mobile notification for payment confirmation",
            "Please add ability to tag and categorize tickets",
            "Feature suggestion weekly summary email report",
            "Can you add Slack integration for support notifications",
            "Please allow export to PDF format in reports section",
            "Would love batch operations for managing multiple accounts",
            "Feature request advanced filtering in ticket search",
        ],
    ),
    IntentTemplate(
        intent="subscription_cancel",
        priority="high",
        department="Retention Team",
        templates=[
            "I want to cancel my subscription immediately",
            "Please cancel my plan and stop future billing",
            "How do I unsubscribe from the premium plan",
            "I need to downgrade or cancel my current subscription",
            "Cancel my account I am switching to a competitor",
            "Want to cancel before next billing cycle",
            "Please stop auto renewal of my subscription",
            "I want to cancel the pricing is too high now",
            "Need to cancel subscription due to budget constraints",
            "Cancel my plan but keep my data",
            "I accidentally subscribed to annual plan please cancel",
            "Cancelling because the features I need are not available",
            "Please cancel and confirm via email once done",
            "I no longer need the service please cancel subscription",
            "Cancel immediately and refund unused days of subscription",
        ],
    ),
    IntentTemplate(
        intent="invoice_problem",
        priority="medium",
        department="Finance Team",
        templates=[
            "My invoice has incorrect company name on it",
            "Need GST invoice for my last payment",
            "Invoice not generated for my recent order",
            "The billing address on invoice is wrong",
            "Can you resend the invoice for last month payment",
            "Invoice amount does not match what I was charged",
            "Need a consolidated invoice for last quarter",
            "Tax calculation on invoice appears incorrect",
            "Please update my billing details on the invoice",
            "Invoice PDF is corrupted cannot open it",
            "Missing invoices for last two months",
            "Need invoice with GSTIN for tax filing purposes",
            "Invoice sent to wrong email address",
            "Duplicate invoice generated for same order",
            "Invoice not reflecting the discount that was applied",
        ],
    ),
    IntentTemplate(
        intent="shipping_delay",
        priority="medium",
        department="Logistics Team",
        templates=[
            "My order has not arrived yet it has been many days",
            "Tracking shows shipped but no movement since days",
            "Package is stuck at the warehouse need update",
            "Delivery was promised by a date but still not received",
            "Order shows delivered but I have not received it",
            "Tracking number not working cannot track my package",
            "Package marked as delivered to wrong address",
            "Shipment delayed due to weather but no new ETA given",
            "My order is showing in transit for over a week",
            "Delivery partner attempted delivery but I was home",
            "Package returned to seller without delivery attempt",
            "Express delivery ordered but showing standard shipping",
            "Order split into multiple packages but only one arrived",
            "International shipment stuck at customs",
            "Received wrong item in shipment correct order still pending",
        ],
    ),
    IntentTemplate(
        intent="general_inquiry",
        priority="low",
        department="General Support",
        templates=[
            "What are your customer support hours",
            "How do I update my profile information",
            "Where can I find my account settings",
            "What payment methods do you accept",
            "How do I change my password",
            "Can I have multiple users on one account",
            "What is your refund policy",
            "How long does it take to process a refund",
            "Do you offer discounts for annual subscription",
            "How do I add a new team member to my workspace",
            "What browsers are officially supported",
            "Is there a mobile app available for Android",
            "How do I download my data from the platform",
            "What is the difference between free and paid plans",
            "How do I contact support for urgent issues",
        ],
    ),
]

NOISE_PREFIXES = [
    "", "", "", "", "",
    "Hi, ", "Hello, ", "Please help, ",
    "Urgent: ", "HELP: ",
]

NOISE_SUFFIXES = [
    "", "", "", "", "",
    " Please help.", " Need urgent help.", "!",
    " ASAP please.", " Thanks.",
]


def add_noise(text):
    prefix = random.choice(NOISE_PREFIXES)
    suffix = random.choice(NOISE_SUFFIXES)
    if random.random() < 0.3:
        text = text.lower()
    return prefix + text + suffix


def generate_dataset(samples_per_intent=200, output_path="data/raw/customer_support_dataset.csv"):
    records = []

    for intent_template in INTENT_TEMPLATES:
        templates = intent_template.templates
        count = 0
        while count < samples_per_intent:
            template = random.choice(templates)
            text = add_noise(template)
            if len(text.strip()) < 5:
                continue
            records.append({
                "text": text,
                "intent": intent_template.intent,
                "priority": intent_template.priority,
                "department": intent_template.department,
            })
            count += 1

    random.shuffle(records)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent", "priority", "department"])
        writer.writeheader()
        writer.writerows(records)

    print("\n Dataset generated successfully!")
    print(f" Total samples : {len(records)}")
    print(f" Intents       : {len(INTENT_TEMPLATES)}")
    print(f" Per intent    : {samples_per_intent}")
    print(f" Output        : {output_path}")
    print("\n Class distribution:")
    counts = {}
    for r in records:
        counts[r["intent"]] = counts.get(r["intent"], 0) + 1
    for intent, cnt in sorted(counts.items()):
        print(f"   {intent}: {cnt}")


if __name__ == "__main__":
    generate_dataset(samples_per_intent=200)
