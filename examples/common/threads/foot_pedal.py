"""Foot pedal reader thread – evdev/pynput listener that fires button callbacks."""

import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from common.data_manager import DataManager

# Add workspace roots to path for standalone testing
_examples_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_examples_path))  # Add 'examples' folder to find 'common'
sys.path.insert(0, str(_examples_path.parent))  # Add 'example_agilex' folder


class FootPedal:
    """Foot pedal reader – fires task-agnostic button callbacks on key press.

    Assign callables to ``on_button_a``, ``on_button_b``, ``on_button_c``
    then call ``run()`` in a daemon thread::

        pedal = FootPedal(
            data_manager,
            {"button_a": "a", "button_b": "b", "button_c": "c"},
        )
        pedal.on_button_a = my_fn
        threading.Thread(target=pedal.run, daemon=True).start()
    """

    def __init__(
        self,
        data_manager: DataManager,
        key_map: dict[str, Any],
    ) -> None:
        """Initialize FootPedal.

        Args:
            data_manager: Shared state manager (used for shutdown signalling).
            key_map: Mapping of button names to key chars, e.g.
                ``{"button_a": "a", "button_b": "b", "button_c": "c"}``.
        """
        self._data_manager = data_manager
        self._key_map = key_map

        self.on_button_a: Callable[[], None] | None = None
        self.on_button_b: Callable[[], None] | None = None
        self.on_button_c: Callable[[], None] | None = None

    def _dispatch(self, char: str) -> None:
        """Fire the matching callback for *char*."""
        if char == self._key_map.get("button_a") and self.on_button_a:
            self.on_button_a()
        elif char == self._key_map.get("button_b") and self.on_button_b:
            self.on_button_b()
        elif char == self._key_map.get("button_c") and self.on_button_c:
            self.on_button_c()

    def run(self) -> None:
        """Block and dispatch key events until shutdown is requested."""
        print(f"⌨️  Foot pedal listener started. Mappings: {self._key_map}")

        # -- evdev (preferred on Linux, exclusive grab) ----------------------
        try:
            import evdev  # type: ignore[import]

            devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
            pedals = [
                d for d in devices if "PCsensor" in d.name or "FootSwitch" in d.name
            ]

            if pedals:
                dev = next((p for p in pedals if "Keyboard" in p.name), pedals[0])
                print(f"⌨️  Foot pedal acquired via evdev: {dev.name}")
                try:
                    dev.grab()
                    for event in dev.read_loop():
                        if self._data_manager.is_shutdown_requested():
                            break
                        if event.type == evdev.ecodes.EV_KEY:
                            k = evdev.categorize(event)
                            if k.keystate == k.key_down:
                                key_str = k.keycode
                                if isinstance(key_str, list):
                                    key_str = key_str[0]
                                char = key_str.replace("KEY_", "").lower()
                                print(f"🔍 [PEDAL] Key: '{char}'")
                                self._dispatch(char)
                except Exception as e:
                    print(f"⚠️  evdev read error: {e}")
                finally:
                    try:
                        dev.ungrab()
                    except Exception:
                        pass
                print("⌨️  Foot pedal stopped (evdev).")
                return

        except Exception as e:
            print(f"⚠️  evdev unavailable: {e} — falling back to pynput")

        # -- pynput fallback ------------------------------------------------
        try:
            from pynput import keyboard  # type: ignore[import]

            print("⌨️  Foot pedal listener (pynput fallback) started.")

            def on_press(key: object) -> None:
                """Forward key press."""
                try:
                    char = key.char if hasattr(key, "char") else str(key)  # type: ignore[union-attr]
                    self._dispatch(char)
                except Exception:
                    pass

            with keyboard.Listener(on_press=on_press) as listener:
                while not self._data_manager.is_shutdown_requested():
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
    # -- Standalone Hardware Test -------------------------------------------
    from common.data_manager import DataManager

    print("🚀 Standalone Foot Pedal Hardware Test")
    print("--------------------------------------")
    dm = DataManager()
    # Default mappings
    mappings = {"button_a": "a", "button_b": "b", "button_c": "c"}
    pedal = FootPedal(dm, mappings)

    pedal.on_button_a = lambda: print("✓ 🟡 Pedal A (ENABLE/DISABLE) detected")
    pedal.on_button_b = lambda: print("✓ 🏠 Pedal B (HOME) detected")
    pedal.on_button_c = lambda: print("✓ 🔴 Pedal C (RECORD) detected")

    try:
        pedal.run()
    except KeyboardInterrupt:
        print("\n👋 Test stopped.")
