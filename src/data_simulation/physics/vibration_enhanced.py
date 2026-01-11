"""
Enhanced Vibration Signal Generation with Envelope Modulation

This module implements realistic vibration signatures for rotating equipment
with amplitude-modulated bearing defect patterns suitable for envelope analysis.

Key Features:
- Amplitude modulation (envelope) for bearing defects
- Structural resonance modeling
- Bearing geometry-based defect frequencies
- Progression from incipient to severe defects

Reference: ISO 10816, Randall "Vibration-based Condition Monitoring"
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BearingGeometry:
    """Physical bearing geometry for defect frequency calculation."""
    n_balls: int = 9              # Number of rolling elements
    ball_diameter: float = 12.0   # Ball diameter (mm)
    pitch_diameter: float = 60.0  # Pitch diameter (mm)
    contact_angle: float = 0.0    # Contact angle (radians)

    def calculate_defect_frequencies(self, f_shaft: float) -> Dict[str, float]:
        """
        Calculate bearing defect frequencies based on geometry.

        Args:
            f_shaft: Shaft rotation frequency (Hz)

        Returns:
            Dict with BPFO, BPFI, BSF, FTF frequencies
        """
        bd_pd = self.ball_diameter / self.pitch_diameter
        cos_phi = np.cos(self.contact_angle)

        # Ball Pass Frequency Outer race
        bpfo = (self.n_balls / 2) * (1 - bd_pd * cos_phi) * f_shaft

        # Ball Pass Frequency Inner race
        bpfi = (self.n_balls / 2) * (1 + bd_pd * cos_phi) * f_shaft

        # Ball Spin Frequency
        bsf = (self.pitch_diameter / (2 * self.ball_diameter)) * \
            (1 - (bd_pd * cos_phi)**2) * f_shaft

        # Fundamental Train Frequency (cage)
        ftf = 0.5 * (1 - bd_pd * cos_phi) * f_shaft

        return {
            'bpfo': bpfo,
            'bpfi': bpfi,
            'bsf': bsf,
            'ftf': ftf
        }


class EnhancedVibrationGenerator:
    """
    Generates realistic vibration signals with envelope-modulated bearing defects.
    """

    def __init__(self,
                 sample_rate: int = 10240,
                 resonance_freq: float = 3000,
                 bearing_geometry: Optional[BearingGeometry] = None):
        """
        Initialize vibration generator.

        Args:
            sample_rate: Samples per second (Hz)
            resonance_freq: Structural resonance frequency (Hz)
            bearing_geometry: Bearing physical parameters
        """
        self.sample_rate = sample_rate
        self.resonance_freq = resonance_freq
        self.bearing_geometry = bearing_geometry or BearingGeometry()
        self._phase = 0.0

    def generate_bearing_vibration(self,
                                   rpm: float,
                                   bearing_health: float,
                                   duration: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Generate vibration signal with bearing defect signatures.

        Args:
            rpm: Rotor speed (RPM)
            bearing_health: Health indicator (1.0 = healthy, 0.0 = failed)
            duration: Signal duration (seconds)

        Returns:
            Tuple of (signal, metrics_dict)
        """
        if rpm <= 0:
            n_samples = int(self.sample_rate * duration)
            return np.random.normal(0, 0.05, n_samples), {'rms': 0.05, 'peak': 0.15}

        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        f_shaft = rpm / 60.0  # Hz

        # Calculate defect frequencies
        defect_freqs = self.bearing_geometry.calculate_defect_frequencies(f_shaft)

        # Initialize signal
        signal = np.zeros(n_samples)

        # 1. Healthy baseline vibration (synchronous components)
        signal += self._generate_healthy_baseline(t, f_shaft)

        # 2. Add bearing defect signatures based on health degradation
        if bearing_health < 0.95:
            signal += self._generate_outer_race_defect(
                t, defect_freqs['bpfo'], bearing_health
            )

        if bearing_health < 0.75:
            signal += self._generate_inner_race_defect(
                t, defect_freqs['bpfi'], bearing_health
            )

        if bearing_health < 0.6:
            signal += self._generate_ball_defect(
                t, defect_freqs['bsf'], bearing_health
            )

        if bearing_health < 0.5:
            # Severe degradation - add broadband noise
            signal += self._generate_degradation_noise(bearing_health, n_samples)

        # 3. Add measurement noise
        signal += np.random.normal(0, 0.08, n_samples)

        # Update phase for continuity
        self._phase = (self._phase + 2 * np.pi * f_shaft * duration) % (2 * np.pi)

        # Compute metrics
        metrics = self._compute_vibration_metrics(signal, defect_freqs, f_shaft)

        return signal, metrics

    def _generate_healthy_baseline(self, t: np.ndarray, f_shaft: float) -> np.ndarray:
        """Generate healthy machine vibration signature."""
        signal = np.zeros_like(t)

        # 1X synchronous component (dominant)
        signal += 0.4 * np.sin(2 * np.pi * f_shaft * t + self._phase)

        # 2X component (slight unbalance)
        signal += 0.15 * np.sin(2 * np.pi * 2 * f_shaft * t)

        # 3X component (minor misalignment)
        signal += 0.05 * np.sin(2 * np.pi * 3 * f_shaft * t)

        return signal

    def _generate_outer_race_defect(self,
                                    t: np.ndarray,
                                    f_bpfo: float,
                                    health: float) -> np.ndarray:
        """
        Generate outer race defect with envelope modulation.

        Outer race defects develop first (stationary race under load).
        Uses amplitude modulation to simulate impact-excited resonance.
        """
        # Degradation severity (0.0 to 1.0)
        severity = min(1.0, (0.95 - health) / 0.95)

        # Modulation depth increases with severity
        modulation_index = severity * 0.7

        # Carrier: structural resonance
        carrier = np.sin(2 * np.pi * self.resonance_freq * t)

        # Envelope: bearing defect frequency
        envelope = 1 + modulation_index * np.sin(2 * np.pi * f_bpfo * t)

        # Amplitude scales with severity
        amplitude = severity * 1.5

        # Amplitude-modulated signal
        signal = amplitude * envelope * carrier

        # Add sidebands (modulation creates frequency sidebands)
        f_shaft = f_bpfo / self.bearing_geometry.n_balls * 2  # Approximate
        signal += amplitude * 0.3 * np.sin(2 * np.pi * (self.resonance_freq + f_bpfo) * t)
        signal += amplitude * 0.3 * np.sin(2 * np.pi * (self.resonance_freq - f_bpfo) * t)

        return signal

    def _generate_inner_race_defect(self,
                                    t: np.ndarray,
                                    f_bpfi: float,
                                    health: float) -> np.ndarray:
        """
        Generate inner race defect signature.

        Inner race defects show amplitude modulation at shaft frequency
        (rotating race creates load zone variation).
        """
        severity = min(1.0, (0.75 - health) / 0.75)

        # Inner race defects are amplitude-modulated by shaft rotation
        f_shaft = f_bpfi / self.bearing_geometry.n_balls * 2  # Approximate

        # Double modulation: defect frequency modulated by shaft frequency
        modulation_index_defect = severity * 0.6
        modulation_index_shaft = severity * 0.4

        carrier = np.sin(2 * np.pi * self.resonance_freq * t)
        envelope_defect = 1 + modulation_index_defect * np.sin(2 * np.pi * f_bpfi * t)
        envelope_shaft = 1 + modulation_index_shaft * np.sin(2 * np.pi * f_shaft * t)

        amplitude = severity * 2.0
        signal = amplitude * envelope_defect * envelope_shaft * carrier

        return signal

    def _generate_ball_defect(self,
                             t: np.ndarray,
                             f_bsf: float,
                             health: float) -> np.ndarray:
        """Generate ball (rolling element) defect signature."""
        severity = min(1.0, (0.6 - health) / 0.6)

        # Ball defects create random impacts (multiple balls)
        amplitude = severity * 1.2

        # Lower frequency modulation
        carrier = np.sin(2 * np.pi * self.resonance_freq * t)
        envelope = 1 + severity * 0.5 * np.sin(2 * np.pi * f_bsf * t)

        signal = amplitude * envelope * carrier

        return signal

    def _generate_degradation_noise(self, health: float, n_samples: int) -> np.ndarray:
        """
        Generate broadband noise for severe bearing degradation.

        Advanced wear increases random vibration across all frequencies.
        """
        severity = (0.5 - health) / 0.5
        noise_level = severity * 0.8

        # White noise with increasing amplitude
        noise = np.random.normal(0, noise_level, n_samples)

        return noise

    def _compute_vibration_metrics(self,
                                   signal: np.ndarray,
                                   defect_freqs: Dict,
                                   f_shaft: float) -> Dict:
        """Compute standard vibration metrics."""
        # RMS velocity (primary metric)
        rms = np.sqrt(np.mean(signal**2))

        # Peak velocity
        peak = np.max(np.abs(signal))

        # Crest factor (peak/rms - increases with defects)
        crest_factor = peak / rms if rms > 0 else 0

        # Kurtosis (increases with impulsive defects)
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            kurtosis = np.mean(((signal - mean) / std)**4)
        else:
            kurtosis = 0

        return {
            'rms': round(rms, 4),
            'peak': round(peak, 4),
            'crest_factor': round(crest_factor, 3),
            'kurtosis': round(kurtosis, 3),
            'bpfo_freq': round(defect_freqs['bpfo'], 2),
            'bpfi_freq': round(defect_freqs['bpfi'], 2)
        }

if __name__ == '__main__':
    """Demonstration and validation."""
    import matplotlib.pyplot as plt

    print("Enhanced Vibration Generator - Demonstration")

    # Create generator with custom bearing geometry
    bearing = BearingGeometry(n_balls=9, ball_diameter=12.0, pitch_diameter=60.0)
    gen = EnhancedVibrationGenerator(sample_rate=10240, bearing_geometry=bearing)

    # Test different health states
    health_states = [0.95, 0.75, 0.55, 0.35]
    rpm = 3000

    fig, axes = plt.subplots(len(health_states), 1, figsize=(12, 10))

    for i, health in enumerate(health_states):
        signal, metrics = gen.generate_bearing_vibration(rpm, health, duration=0.5)

        axes[i].plot(signal[:2048], linewidth=0.5)
        axes[i].set_title(f"Health={health:.2f} | RMS={metrics['rms']:.2f} mm/s | "
                         f"Crest={metrics['crest_factor']:.2f} | Kurt={metrics['kurtosis']:.2f}")
        axes[i].set_ylabel('Velocity (mm/s)')
        axes[i].grid(True, alpha=0.3)

        print(f"\nHealth {health:.2f}:")
        print(f"  RMS: {metrics['rms']:.3f} mm/s")
        print(f"  Peak: {metrics['peak']:.3f} mm/s")
        print(f"  Crest Factor: {metrics['crest_factor']:.2f}")
        print(f"  Kurtosis: {metrics['kurtosis']:.2f}")
        print(f"  BPFO: {metrics['bpfo_freq']:.2f} Hz")
        print(f"  BPFI: {metrics['bpfi_freq']:.2f} Hz")

    axes[-1].set_xlabel('Sample')
    plt.tight_layout()
    plt.savefig('vibration_degradation_demo.png', dpi=150)
    print("\nPlot saved to: vibration_degradation_demo.png")