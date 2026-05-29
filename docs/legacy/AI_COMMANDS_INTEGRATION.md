# AI Commands Integration Guide

## Overview
The frontend has been successfully integrated with the backend's LLM modules. Users can now trigger AI-powered features directly from the chat interface using special commands.

## Available Commands

### 1. **@gro analyze** - Inventory Analysis
Analyzes current inventory levels and identifies items with low stock.

**Trigger:** Type `@gro analyze` in the chat
**Backend Module:** `inventory_analyzer.py`
**Returns:** 
- Low stock items with current vs. safety stock
- Healthy stock items
- Warnings for items below safety levels

**Display:** Dialog shows items in categorized lists with color coding
- 🔴 Red for low stock warnings
- 🟢 Green for healthy items

---

### 2. **@gro menu** - Menu Suggestions
Generates menu suggestions based on available inventory.

**Trigger:** Type `@gro menu` in the chat
**Backend Module:** `menu_generator.py`
**Returns:** 
- Recommended dishes
- Ingredient lists for each dish
- Options that use available inventory efficiently

**Display:** Dialog shows dish names with ingredients organized in cards

---

### 3. **@gro restock** - Restock Planning
Recommends items to restock and suggests quantities based on inventory history.

**Trigger:** Type `@gro restock` in the chat
**Backend Module:** `procurement_planner.py`
**Returns:**
- Recommended items with quantities
- Pricing information
- Optimal timing for purchases

**Display:** Dialog shows items with suggested quantities and prices

---

### 4. **@gro plan** - Procurement Planning
Creates a comprehensive procurement plan with group assignment and timeline.

**Trigger:** Type `@gro plan` in the chat
**Backend Module:** `chat_procurement_planner.py`
**Returns:**
- Items to procure
- Assignment to group members
- Timeline for procurement activities
- Coordinated shopping plan

**Display:** Dialog shows items, assigned members, and execution timeline

---

## How It Works

1. **User Input:** User types a command (e.g., `@gro analyze`) in the chat
2. **Message Sent:** Message is sent via HTTP POST to `/api/rooms/{roomId}/messages`
3. **Backend Processing:** Backend's `maybe_answer_with_llm()` detects the command
4. **LLM Execution:** Appropriate LLM module processes the request
5. **WebSocket Broadcast:** Backend sends `ai_event` via WebSocket connection
6. **Frontend Parsing:** Frontend's `_handleWebSocketMessage()` receives the event
7. **UI Display:** Appropriate content builder renders the results in a dialog

## Technical Integration

### Frontend Components Added:

**File: `lib/models/ai_event.dart`**
- `AIEvent` class for parsing WebSocket messages
- Helper methods: `isInventoryEvent`, `isMenuEvent`, `isRestockEvent`, `isProcurementEvent`
- Factory constructor: `AIEvent.fromJson()` for message deserialization

**File: `lib/pages/chat_detail_page.dart`**
- Updated `_handleWebSocketMessage()` to recognize `ai_event` type messages
- Added `_showAIEventDialog()` - Main dialog display method
- Added 4 content builder methods:
  - `_buildInventoryContent()` - Renders inventory analysis
  - `_buildMenuContent()` - Renders menu suggestions
  - `_buildRestockContent()` - Renders restock recommendations
  - `_buildProcurementContent()` - Renders procurement plan
- Updated chat input placeholder with command hints

### WebSocket Event Format:
```json
{
  "type": "ai_event",
  "event": "inventory_analysis|menu|restock_plan|procurement_plan",
  "room_id": 123,
  "narrative": "Human-readable explanation of results",
  "payload": {
    // Event-specific data structure
  }
}
```

## Testing the Integration

### Prerequisites:
- Backend API running with new AI modules
- WebSocket connection established
- Valid authentication token

### Test Steps:
1. Open chat with a room/group
2. Type `@gro analyze` and hit send
3. Wait for response (may take 5-10 seconds for LLM processing)
4. A dialog should appear with inventory analysis
5. Repeat with other commands: `@gro menu`, `@gro restock`, `@gro plan`

### Expected Behavior:
- Message appears in chat history
- Brief moment of processing (loading state would be ideal)
- Dialog pops up with formatted results
- User can close dialog and continue chatting

## Troubleshooting

### Dialog doesn't appear:
- Check browser console for WebSocket errors
- Verify backend is running and connected
- Confirm auth token is valid
- Check that room_id is correct

### Empty or malformed data:
- Verify backend LLM modules are working
- Check `/backend/llm_modules/README_AI_MODULES.md` for payload format
- Ensure backend is sending correct event type names

### Command not recognized:
- Check that message starts with exact format: `@gro analyze` (space between command parts)
- Verify backend's `handle_gro_command()` function is configured

## Future Enhancements

1. **Loading Indicator** - Add visual feedback while processing
2. **Command History** - Show recently used commands
3. **Command Autocomplete** - Auto-complete as user types @gro
4. **Result Export** - Allow users to export/share results
5. **Caching** - Cache results to avoid duplicate processing
6. **Error Handling** - Display friendly error messages if LLM fails
7. **@inventory Command** - Add support for inventory management commands
8. **Custom Filters** - Allow filtering results in dialogs

## Files Modified:
- ✅ `/flutter_frontend/lib/models/ai_event.dart` - Created
- ✅ `/flutter_frontend/lib/pages/chat_detail_page.dart` - Updated WebSocket handler and UI methods
- ✅ `/flutter_frontend/lib/services/api_client.dart` - API address already correct

## Backend Integration:
- ✅ Backend already has all required modules
- ✅ `/backend/app.py` - `broadcast_ai_event()` function ready
- ✅ `/backend/llm_modules/` - All 6 LLM modules available
- ✅ No backend changes required for this integration
