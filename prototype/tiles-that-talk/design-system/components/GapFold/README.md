# GapFold
Proposed. In a list of windows or days, the valid rows draw as usual and each consecutive run of invalid rows folds into one hatched row naming them and the reason.

- Same gap status as today (valid, ≈, invalid). A ≈ row is valid and never folds.
- The fold is the 135° hatch with `ink` text, `r-sm` corners, 36px tall, so it still reads "not valid" at a glance.
- Runs with different reasons fold separately ("gap in record" is not "before the archive").
- Applies to the periods chart, the ten-day list and the rolling-windows table.
- The spoken reason per row stays in screen-reader text.
