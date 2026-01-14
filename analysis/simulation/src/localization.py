"""
3D Sound Source Localization using Multilateration.

Given TDOA measurements between microphone pairs, this module
estimates the 3D position of the sound source.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.optimize import least_squares, minimize
import warnings

from .config import SPEED_OF_SOUND
from .microphone import MicrophoneArray
from .tdoa import TDOAEstimate


@dataclass
class LocalizationResult:
    """Result of source localization."""
    position: np.ndarray          # Estimated (x, y, z) position
    position_error: Optional[float]  # Estimation error if ground truth known
    residual: float               # Optimization residual
    confidence: float             # Overall confidence (0-1)
    method: str                   # Method used for localization
    iterations: int               # Number of optimization iterations
    covariance: Optional[np.ndarray] = None  # Position uncertainty


class Multilaterator:
    """
    3D source localization using multilateration.

    Multilateration uses the difference in distances from the source
    to multiple receivers (derived from TDOA) to determine position.
    """

    def __init__(self,
                 mic_array: MicrophoneArray,
                 speed_of_sound: float = SPEED_OF_SOUND):
        """
        Initialize the multilaterator.

        Args:
            mic_array: Microphone array with known positions
            speed_of_sound: Speed of sound in m/s
        """
        self.mic_array = mic_array
        self.speed_of_sound = speed_of_sound
        self.mic_positions = mic_array.positions

    def _tdoa_residual(self,
                      position: np.ndarray,
                      tdoa_estimates: List[TDOAEstimate]) -> np.ndarray:
        """
        Calculate residuals between measured and predicted TDOA.

        Args:
            position: Candidate source position (x, y, z)
            tdoa_estimates: List of TDOA measurements

        Returns:
            Array of residuals weighted by confidence
        """
        residuals = []

        for tdoa in tdoa_estimates:
            # Get microphone positions
            pos_i = self.mic_array.get_microphone(tdoa.mic_i).position
            pos_j = self.mic_array.get_microphone(tdoa.mic_j).position

            # Calculate predicted distances
            dist_i = np.linalg.norm(position - pos_i)
            dist_j = np.linalg.norm(position - pos_j)

            # Predicted TDOA
            predicted_tdoa = (dist_i - dist_j) / self.speed_of_sound

            # Residual weighted by confidence
            residual = (predicted_tdoa - tdoa.tdoa_seconds) * tdoa.confidence
            residuals.append(residual)

        return np.array(residuals)

    def _tdoa_cost(self,
                  position: np.ndarray,
                  tdoa_estimates: List[TDOAEstimate]) -> float:
        """
        Calculate total cost (sum of squared residuals).
        """
        residuals = self._tdoa_residual(position, tdoa_estimates)
        return np.sum(residuals ** 2)

    def _get_initial_guess(self, tdoa_estimates: List[TDOAEstimate]) -> np.ndarray:
        """
        Generate initial position guess for optimization.

        Uses a simple closed-form approximation based on the
        microphone centroid and dominant TDOA directions.
        """
        # Start from centroid
        centroid = self.mic_array.centroid()

        # Refine using strongest TDOA
        if tdoa_estimates:
            # Find TDOA with highest confidence
            best_tdoa = max(tdoa_estimates, key=lambda t: t.confidence)

            pos_i = self.mic_array.get_microphone(best_tdoa.mic_i).position
            pos_j = self.mic_array.get_microphone(best_tdoa.mic_j).position

            # Direction from j to i
            direction = pos_i - pos_j
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Offset from centroid based on TDOA sign
            offset_magnitude = 20.0  # Initial offset in meters
            if best_tdoa.tdoa_seconds > 0:
                # Source closer to mic_j
                centroid = centroid - offset_magnitude * direction
            else:
                # Source closer to mic_i
                centroid = centroid + offset_magnitude * direction

        return centroid

    def localize(self,
                tdoa_estimates: List[TDOAEstimate],
                initial_guess: np.ndarray = None,
                bounds: Tuple[np.ndarray, np.ndarray] = None,
                method: str = 'least_squares') -> LocalizationResult:
        """
        Localize sound source from TDOA measurements.

        Args:
            tdoa_estimates: List of TDOA estimates between mic pairs
            initial_guess: Starting position for optimization
            bounds: Position bounds ((x_min, y_min, z_min), (x_max, y_max, z_max))
            method: 'least_squares' or 'minimize'

        Returns:
            LocalizationResult with estimated position
        """
        if len(tdoa_estimates) < 3:
            warnings.warn("Less than 3 TDOA estimates - localization may be unreliable")

        # Initial guess
        if initial_guess is None:
            initial_guess = self._get_initial_guess(tdoa_estimates)

        # Set default bounds based on array geometry
        if bounds is None:
            coverage = self.mic_array.coverage_volume()
            margin = 100  # meters beyond mic array
            bounds = (
                np.array([
                    coverage['x_range'][0] - margin,
                    coverage['y_range'][0] - margin,
                    0  # Keep z positive (above ground)
                ]),
                np.array([
                    coverage['x_range'][1] + margin,
                    coverage['y_range'][1] + margin,
                    coverage['z_range'][1] + margin
                ])
            )

        if method == 'least_squares':
            result = least_squares(
                self._tdoa_residual,
                initial_guess,
                args=(tdoa_estimates,),
                bounds=bounds,
                method='trf',
                loss='soft_l1',  # Robust to outliers
                max_nfev=1000
            )
            position = result.x
            residual = result.cost
            iterations = result.nfev

            # Estimate covariance from Jacobian
            try:
                J = result.jac
                cov = np.linalg.inv(J.T @ J) * residual / (len(tdoa_estimates) - 3)
            except np.linalg.LinAlgError:
                cov = None

        else:  # minimize
            result = minimize(
                self._tdoa_cost,
                initial_guess,
                args=(tdoa_estimates,),
                method='L-BFGS-B',
                bounds=list(zip(bounds[0], bounds[1])),
                options={'maxiter': 1000}
            )
            position = result.x
            residual = result.fun
            iterations = result.nit
            cov = None

        # Calculate confidence based on residual and number of estimates
        avg_confidence = np.mean([t.confidence for t in tdoa_estimates])
        residual_factor = np.exp(-residual * 100)  # Lower residual = higher confidence
        confidence = avg_confidence * residual_factor

        return LocalizationResult(
            position=position,
            position_error=None,
            residual=residual,
            confidence=min(1.0, confidence),
            method=method,
            iterations=iterations,
            covariance=cov
        )

    def localize_with_grid_search(self,
                                  tdoa_estimates: List[TDOAEstimate],
                                  grid_resolution: float = 10.0) -> LocalizationResult:
        """
        Localize using grid search followed by refinement.

        More robust for cases where local minima might trap optimization.

        Args:
            tdoa_estimates: TDOA measurements
            grid_resolution: Grid spacing in meters

        Returns:
            LocalizationResult
        """
        coverage = self.mic_array.coverage_volume()
        margin = 50

        # Create search grid
        x_range = np.arange(
            coverage['x_range'][0] - margin,
            coverage['x_range'][1] + margin,
            grid_resolution
        )
        y_range = np.arange(
            coverage['y_range'][0] - margin,
            coverage['y_range'][1] + margin,
            grid_resolution
        )
        z_range = np.arange(
            max(1, coverage['z_range'][0] - 10),
            coverage['z_range'][1] + 30,
            grid_resolution
        )

        # Coarse grid search
        best_cost = np.inf
        best_position = self.mic_array.centroid()

        for x in x_range:
            for y in y_range:
                for z in z_range:
                    position = np.array([x, y, z])
                    cost = self._tdoa_cost(position, tdoa_estimates)
                    if cost < best_cost:
                        best_cost = cost
                        best_position = position

        # Refine with optimization
        return self.localize(
            tdoa_estimates,
            initial_guess=best_position,
            method='least_squares'
        )


class IterativeLocalizer:
    """
    Iterative localization that refines estimates over multiple calls.

    Useful for tracking moving sources.
    """

    def __init__(self, mic_array: MicrophoneArray):
        self.multilaterator = Multilaterator(mic_array)
        self.position_history: List[np.ndarray] = []
        self.kalman_state: Optional[np.ndarray] = None

    def update(self, tdoa_estimates: List[TDOAEstimate]) -> LocalizationResult:
        """
        Update position estimate with new TDOA measurements.

        Uses previous position as initial guess for faster convergence.
        """
        if self.position_history:
            initial_guess = self.position_history[-1]
        else:
            initial_guess = None

        result = self.multilaterator.localize(tdoa_estimates, initial_guess)
        self.position_history.append(result.position)

        return result

    def get_smoothed_position(self, window: int = 5) -> np.ndarray:
        """Get smoothed position from recent history."""
        if not self.position_history:
            return None

        recent = self.position_history[-window:]
        return np.mean(recent, axis=0)

    def reset(self):
        """Reset position history."""
        self.position_history = []
        self.kalman_state = None
