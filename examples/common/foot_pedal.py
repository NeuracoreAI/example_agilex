"""Foot pedal abstraction with callback dispatch and key mapping."""

import os
import sys
import time
import traceback
from typing import Callable

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data_manager import DataManager


class FootPedal:
    """Task-agnostic pedal/keyboard mapping with dynamic button support."""

    def __init__(self, key_map: dict[str, str] | None = None) -> None:
        """Initialize dynamic button mapping and callback registry."""
        source = key_map if key_map is not None else {}
        self._key_map = {name: str(key).strip().lower() for name, key in source.items()}
        self._callbacks: dict[str, Callable[[], None]] = {}

    @property
    def key_map(self) -> dict[str, str]:
        """Return a copy of the current button-to-key map."""
        return dict(self._key_map)

    def set_key_map(self, key_map: dict[str, str]) -> None:
        """Set the key map for the foot pedal. Will override the current key map.

        Args:
            key_map: The key map to set for the foot pedal.
        """
        self._key_map = {
            name: str(key).strip().lower() for name, key in key_map.items()
        }

    def bind(self, button_name: str, callback: Callable[[], None] | None) -> None:
        """Bind a callback to a button name.

        Args:
            button_name: The name of the button to bind the callback to.
            callback: The callback to bind to the button name.

        NOTE: if callback is None, the callback is removed from the button name.
        """
        if callback is None:
            self._callbacks.pop(button_name, None)
            return
        self._callbacks[button_name] = callback

    def get_bound_buttons(self) -> list[str]:
        """Return button names that currently have callbacks bound."""
        return list(self._callbacks.keys())

    def _dispatch(self, char: str) -> None:
        """Fire mapped callback for a pressed key."""
        normalized = str(char).strip().lower()
        for button_name, mapped_key in self._key_map.items():
            if normalized == mapped_key:
                callback = self._callbacks.get(button_name)
                if callback:
                    callback()

    def run_loop(self, data_manager: DataManager) -> None:
        """Block and dispatch pedal events until shutdown is requested."""
        print("⌨️  Foot pedal listener started.")

        try:
            from pynput import keyboard

            print("⌨️  Foot pedal listener (pynput) started.")

            def on_press(key: object) -> None:
                try:
                    char = key.char if hasattr(key, "char") else str(key)
                    self._dispatch(char)
                except Exception:
                    pass

            with keyboard.Listener(on_press=on_press) as listener:
                while not data_manager.is_shutdown_requested():
                    if not listener.is_alive():
                        break
                    time.sleep(0.1)
                listener.stop()

        except Exception as e:
            print(f"✗ Fatal error in foot pedal: {e}")
            traceback.print_exc()
        finally:
            print("⌨️  Foot pedal listener stopped.")


if __name__ == "__main__":
    print("🚀 Standalone Foot Pedal Hardware Test")
    print("--------------------------------------")
    data_manager = DataManager()
    pedal = FootPedal({"button_a": "a", "button_b": "b", "button_c": "c"})
    pedal.bind("button_a", lambda: print("✓ 🟡 Pedal A (ENABLE/DISABLE) detected"))
    pedal.bind("button_b", lambda: print("✓ 🏠 Pedal B (HOME) detected"))
    pedal.bind("button_c", lambda: print("✓ 🔴 Pedal C (RECORD) detected"))

    try:
        pedal.run_loop(data_manager)
    except KeyboardInterrupt:
        print("\n👋 Test stopped.")
