/**
 * The rain intensity steps a bar's height tells apart (DESIGN.md, "The Hue
 * Is Never Alone Rule": rain intensity is bar height, not shade). The
 * Overview hero's quarter-hour rain bars and the Rain tab's hourly bars use
 * the one rule.
 *
 * Offers `rainStep` and `RAIN_STEP_COUNT`.
 */

/** Rain-rate ceilings (in/hr) of the first three steps; above is step 4. */
const RAIN_STEPS = [0.05, 0.1, 0.3] as const;

/** How many steps there are: light, moderate, steady, heavy. */
export const RAIN_STEP_COUNT = 4;

/**
 * The rain-ramp step for one slot's rain, by its rate.
 *
 * @param rainIn - The rain that fell in the slot.
 * @param slotMinutes - The slot's length.
 * @returns 1 (up to 0.05 in/hr) to 4 (over 0.3 in/hr), or null for no rain.
 */
export function rainStep(rainIn: number | null, slotMinutes = 15): 1 | 2 | 3 | 4 | null {
  if (rainIn === null || rainIn <= 0) return null;
  const rate = (rainIn * 60) / slotMinutes;
  const step = RAIN_STEPS.findIndex((ceiling) => rate <= ceiling + 1e-9);
  return (step === -1 ? 4 : step + 1) as 1 | 2 | 3 | 4;
}
