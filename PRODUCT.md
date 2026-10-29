# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The station operator: the owner of the "lookout post" station, who checks it several times a day, mostly on a phone. A few curious friends visit occasionally through a shared link. No other audiences.

## Product Purpose

Lookout is the dashboard for one Ambient Weather station. It answers "what is it doing outside right now, and how normal is that?" Success is a glance that gives current conditions in the context of recent history, with deeper views for rain, rain events and solar energy.

## Positioning

Lookout shows this station's own measured, hyperlocal record, which a generic weather app cannot. The archive starts in February 2023. Rain is modelled as discrete rain events in an Event Catalog, and solar energy is tracked over time. Because the archive spans several years, Lookout can make almanac comparisons, such as this month against the same month in earlier years.

## Operating Context

- The owner checks it on a phone, between other things, in short glances.
- The dashboard is public and read-only, with no login.
- A home-screen widget (iOS) is a later goal, fed by a public current-conditions feed.

## Capabilities and Constraints

- **Measured data only.** There is no forecast. Every number is something this station recorded.
- **Recency comparisons.** Rolling windows of the last 7, 30, 90 and 365 days.
- **Cyclic and almanac comparisons.** For example, this month against the same month last year, or today against this date in earlier years.
- **Archive gaps.** Some periods have missing data. A comparison over a gap must be shown as not valid, never silently computed.
- **Tabs.** Overview, Rain, Rain Events and Solar carry over from the Streamlit app. The visualizations may change completely.
- **Persistence.** The Event Catalog keeps its lazy persistence: it saves to Storj automatically when an ongoing event completes. A manual save stays available.
- **Units.** Imperial: °F, mph, inches of rain, inHg, W/m², kWh/m², and the UV index.
- **Timezone.** America/Los_Angeles. Station-local time is the analytical frame.

## Brand Commitments

- **Name:** "Lookout". The station is "lookout post".
- **Binding references:** the presentation of the Dark Sky weather app, and the data storytelling of the Financial Times, where a chart states its finding.

## Evidence on Hand

- `prototype/dashboard/data.json`: a real snapshot of the archive, from 2023-02 to 2026-10, plus one live reading.
- The archive is on Storj, about 374k rows at 5-minute resolution.
- There is no forecast data, and none may be fabricated.

## Product Principles

1. **Lead with now, then context.** Current conditions come first, and recent history explains them.
2. **Promote what is happening.** When it rains, the rain event leads.
3. **Every chart states its finding.** Headlines say what the data shows, not what the chart is.
4. **Honest about the record.** Gaps and invalid comparisons are visible, never smoothed over.
