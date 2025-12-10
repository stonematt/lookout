# Incremental Testing Guidelines for Streamlit Applications

## Overview
This document provides systematic testing approaches for validating Streamlit application changes incrementally.

## Core Principles

### 1. Never Assume Success from Startup Messages
- ❌ Wrong: Seeing "You can now view your Streamlit app" = success
- ✅ Right: Verify actual functionality, not just server binding

### 2. Test in Both Modes
- **Headless mode**: Quick validation, but may miss runtime errors
- **Interactive mode**: Full functionality testing, catches more issues
- **Always test both** before declaring success

### 3. Check Parameters Before Using
```bash
# Always verify API compatibility
python -c "import streamlit as st; help(st.set_page_config)"
```

### 4. Use Multiple Validation Methods
- Console log analysis
- HTTP response checking (when appropriate)
- Error pattern matching
- Functional testing

## Testing Checklist

### Phase 1: Basic Startup (5-10 seconds)
```bash
# Check for immediate errors
timeout 10s streamlit run streamlit_app.py --server.headless true --server.port 8501 > /tmp/test.log 2>&1 & sleep 5

# Check for error patterns
if grep -qi "error\|exception\|traceback\|failed" /tmp/test.log; then
    echo "❌ Startup failed"
    cat /tmp/test.log
    exit 1
fi

# Check for success patterns  
if grep -q "You can now view your Streamlit app" /tmp/test.log; then
    echo "✅ Server started"
else
    echo "❌ Server failed to start"
    cat /tmp/test.log
    exit 1
fi

pkill -f streamlit
```

### Phase 2: Runtime Execution (10-15 seconds)
```bash
# Let app run longer to catch runtime errors
timeout 15s streamlit run streamlit_app.py --server.headless true --server.port 8502 > /tmp/runtime_test.log 2>&1 & sleep 10

# Check for runtime errors
if grep -qi "uncaught\|runtime\|execution" /tmp/runtime_test.log; then
    echo "❌ Runtime errors detected"
    cat /tmp/runtime_test.log
    exit 1
fi

pkill -f streamlit
```

### Phase 3: Interactive Testing (Manual)
```bash
# Run in interactive mode for full testing
echo "Starting interactive mode - test manually in browser"
streamlit run streamlit_app.py --server.port 8503
# Manual checklist:
# - [ ] Page loads without errors
# - [ ] All tabs accessible
# - [ ] Visualizations render correctly
# - [ ] No console errors in browser dev tools
```

## Error Pattern Detection

### Critical Patterns to Check
```bash
# Always check for these patterns
grep -E "(ERROR|Exception|Traceback|TypeError|AttributeError|KeyError|ImportError)" /tmp/test.log

# Common Streamlit issues
grep -E "(unexpected keyword argument|got an unexpected|failed to import)" /tmp/test.log

# Memory/resource issues
grep -E "(MemoryError|OutOfMemoryError|Killed)" /tmp/test.log
```

## Specific Change Types

### Configuration Changes
1. **Verify API compatibility**: Check valid parameters first
2. **Test both headless and interactive**: Different behaviors
3. **Check file creation**: Config files written correctly
4. **Validate functionality**: Not just startup

### Code Logic Changes
1. **Unit test individual functions**: If possible
2. **Integration test**: Full app flow
3. **Edge case testing**: Empty data, error conditions
4. **Performance impact**: Memory usage, startup time

### UI/Theme Changes
1. **Visual verification**: Browser testing required
2. **Cross-browser check**: Different browsers may render differently
3. **Responsive testing**: Different screen sizes
4. **Accessibility**: Contrast, readability

## Common Pitfalls

### False Positives
- Server starts but app crashes later
- Headless mode works, interactive fails
- No immediate errors but functionality broken

### Testing Environment
- Use same Python environment as production
- Check dependencies match target environment
- Verify secrets/config files available

### Timing Issues
- Some errors occur after initial startup
- Race conditions in app initialization
- Async operations not completed

## Regression Testing Strategy

### Before Changes
1. Document current behavior
2. Capture baseline logs
3. Note performance metrics

### After Changes
1. Compare behavior to baseline
2. Look for new error patterns
3. Verify all previous functionality still works

### Quick Validation Commands
```bash
# Comprehensive quick test
function quick_test() {
    local port=$1
    timeout 10s streamlit run streamlit_app.py --server.headless true --server.port $port > /tmp/quick_$port.log 2>&1 & sleep 6

## AI-Specific Testing Requirements

### Archive Data Structure Testing

**AI agents MUST use correct test data structure to avoid `date`/`datetime`/`dateutc` confusion:**

#### Required Test Data Format
```python
def create_test_archive_data():
    """Create test data matching actual archive structure"""
    dates = pd.date_range("2023-01-01", periods=100, freq="5min", tz="America/Los_Angeles")
    return pd.DataFrame({
        "date": dates,  # TZ-aware datetime (PRIMARY)
        "dateutc": [int(dt.timestamp() * 1000) for dt in dates],  # Milliseconds
        "solarradiation": [max(0, 800 * (1 - abs((i % 288) - 144) / 144)) for i in range(100)],
        "tempf": [70 + 10 * (i % 24) / 24 for i in range(100)],
        "dailyrainin": [0.0] * 100,
        "baromrelin": [30.0 + 0.1 * (i % 10) for i in range(100)],
    })
```

#### AI Test Data Validation Checklist
- [ ] Used `date` column for datetime operations (not `datetime`)
- [ ] Included `dateutc` column with millisecond timestamps
- [ ] Set correct timezone on `date` column (`America/Los_Angeles`)
- [ ] Verified column existence checks in code
- [ ] Tested with actual archive data structure before committing

#### Forbidden Test Data Patterns
```python
# ❌ DON'T create test data like this
def wrong_test_data():
    return pd.DataFrame({
        "datetime": dates,  # Column doesn't exist in archive
        "timestamp": dates,  # Wrong column name
        "date": dates.date,  # Loses timezone information
    })
```

### AI Code Generation Validation

**Before committing AI-generated code, verify:**

1. **Column Access**: Uses `df['date']` not `df['datetime']`
2. **Existence Checks**: `if 'column' in df.columns` before access
3. **Timezone Trust**: No redundant conversions of `date` column
4. **Data Types**: Correct handling of datetime64 vs int64 columns
5. **Test Coverage**: Tests with actual archive structure

### Debugging AI-Generated Code

**When AI code fails with archive data:**
```python
# Check what columns actually exist
print("Available columns:", df.columns.tolist())
print("Date column type:", df['date'].dtype if 'date' in df.columns else "MISSING")
print("DateUTC column type:", df['dateutc'].dtype if 'dateutc' in df.columns else "MISSING")

# Verify timezone
if 'date' in df.columns:
    print("Date timezone:", df['date'].dt.tz)
    print("Sample dates:", df['date'].head(3).tolist())
```
    
    # Check for immediate errors
    if grep -qi "error\|exception\|traceback" /tmp/quick_$port.log; then
        echo "❌ Port $port: Errors found"
        return 1
    fi
    
    # Check for startup success
    if grep -q "You can now view your Streamlit app" /tmp/quick_$port.log; then
        echo "✅ Port $port: Started successfully"
        pkill -f streamlit
        return 0
    else
        echo "❌ Port $port: Failed to start"
        cat /tmp/quick_$port.log
        return 1
    fi
}

# Test multiple scenarios
quick_test 8501 || exit 1
```

## Documentation

### Always Document
1. What was tested
2. How it was tested
3. Results observed
4. Limitations of testing

### Test Reports
```markdown
## Test Results

### Environment
- Python version: X.X.X
- Streamlit version: X.X.X  
- Browser: Chrome/Firefox/Safari
- OS: macOS/Linux/Windows

### Tests Performed
- [x] Startup validation
- [x] Runtime error checking
- [x] Interactive functionality
- [x] Visual verification

### Results
- ✅ No startup errors
- ✅ No runtime exceptions
- ✅ All tabs functional
- ❌ Theme not applied (expected behavior)

### Issues Found
- None

### Recommendations
- Deploy with confidence
- Monitor for runtime errors in production
```

## Remember
- **One test is not enough**: Use multiple approaches
- **Headless ≠ Interactive**: Always test both
- **Check parameters**: Verify API compatibility first
- **Document everything**: Future agents need to know what worked