"""Alert management and notification dispatchers (Console, SQLite, Telegram)."""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from agri_drift.data.schema import HealthDecision, HealthState
from agri_drift.storage.database import DatabaseManager

logger = logging.getLogger("agri_drift.alerts")


class AlertManager:
    """Dispatches farmer-facing alerts to configured notification sinks."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        enable_console: bool = True,
        enable_telegram: bool = False,
    ):
        self.db_manager = db_manager
        self.enable_console = enable_console
        self.enable_telegram = enable_telegram

        self.telegram_token = os.environ.get("AGRI_TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.environ.get("AGRI_TELEGRAM_CHAT_ID")
        self._last_alert_time: Dict[str, datetime] = {}

    def dispatch(self, decision: HealthDecision) -> None:
        """Evaluate decision and dispatch alert if sensor is suspected or confirmed faulty."""
        if decision.state not in [HealthState.SUSPECTED_FAULT, HealthState.CONFIRMED_FAULT]:
            return

        # Simple deduplication: suppress duplicate alerts within 1 hour for same sensor
        now = decision.timestamp
        last_t = self._last_alert_time.get(decision.sensor_id)
        if last_t and (now - last_t).total_seconds() < 3600:
            return

        self._last_alert_time[decision.sensor_id] = now

        msg = (
            f"[{decision.state.value}] {decision.explanation} "
            f"(Time: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')})"
        )

        # 1. Console / Logging
        if self.enable_console:
            if decision.state == HealthState.CONFIRMED_FAULT:
                logger.warning(msg)
            else:
                logger.info(msg)

        # 2. Database Record
        if self.db_manager:
            self.db_manager.record_decision(decision)

        # 3. Telegram Adapter (Optional)
        if self.enable_telegram and self.telegram_token and self.telegram_chat_id:
            self._send_telegram(msg)

    def _send_telegram(self, text: str) -> None:
        """Send message via Telegram Bot API using requests."""
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }
            resp = requests.post(url, json=payload, timeout=5.0)
            if resp.status_code != 200:
                logger.error(f"Telegram alert delivery failed with status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to deliver Telegram alert: {e}")
