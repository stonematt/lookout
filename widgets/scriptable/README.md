# Lookout widget for Scriptable

An iOS Home Screen widget, small or medium, that shows one station's current reading against its own history. It draws one of three views, whichever the snapshot's `widget_view` names (see "widget view" in `GLOSSARY.md`):

- **Rain event**: the live event's total against the median event, with its step curve.
- **Dry spell**: the current dry spell against the usual dry spell and the longest on record.
- **Temperature ranges**: now against today, 7, 30 and 365 days.

It reads only the public read functions, `/stations` and `/dashboard?station=<MAC>`, and decides nothing itself: the view, every number and every comparison come from the snapshot. It needs no secrets.

## Install

1. Install Scriptable from the App Store.
2. In Scriptable, add a script named `Lookout` and paste in `Lookout.js`.
3. On the Home Screen, add a Scriptable widget (small or medium), long-press it, choose Edit Widget, and set Script to `Lookout`.
4. Parameter: a station name, such as `Salmon Run`. Case and extra spaces don't matter. Leave it empty for **All stations**, which shows the station with the most going on: a rain event first, then a dry spell, then temperature ranges, the default station breaking a tie. A name that matches no station shows the real names.

Tapping the widget opens that station on the site.

## Change it

The logic above `main` (station matching, the view model and its sentences) runs in Node and is tested by `web/tests/scriptable-widget.test.ts` against the golden snapshots. The drawing needs Scriptable: run the script in the app to preview the medium widget.

After a change, paste the new `Lookout.js` into Scriptable again; the widget does not update itself.
