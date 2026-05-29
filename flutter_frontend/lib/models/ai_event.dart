import 'package:intl/intl.dart';

/// Represents an AI-generated event with structured data for custom UI rendering
class AIEvent {
  final String
  eventType; // "inventory_analysis", "menu", "restock_plan", "procurement_plan"
  final String narrative; // Human-readable description
  final Map<String, dynamic> payload; // Structure-specific data for rendering
  final int roomId;
  final DateTime createdAt;

  AIEvent({
    required this.eventType,
    required this.narrative,
    required this.payload,
    required this.roomId,
    required this.createdAt,
  });

  factory AIEvent.fromJson(Map<String, dynamic> json) => AIEvent(
    eventType: json['event'] ?? json['event_type'] ?? 'unknown',
    narrative: json['narrative'] ?? '',
    payload: json['payload'] ?? json['data'] ?? {},
    roomId: json['room_id'] ?? 0,
    createdAt: DateTime.now(),
  );

  String get formattedTime {
    return DateFormat('HH:mm').format(createdAt);
  }

  /// Check if this event is an inventory-related event
  bool get isInventoryEvent => eventType == 'inventory_analysis';

  /// Check if this event is a menu-related event
  bool get isMenuEvent => eventType == 'menu_suggestions';

  /// Check if this event is a restock-related event
  bool get isRestockEvent => eventType == 'restock_plan';

  /// Check if this event is a procurement plan
  bool get isProcurementEvent => eventType == 'procurement_plan';

  // Helpers for payload access
  List<dynamic> get lowStockItems => payload['low_stock'] ?? [];
  List<dynamic> get healthyItems => payload['healthy'] ?? [];
  List<dynamic> get dishes => payload['dishes'] ?? [];
  List<dynamic> get restockItems => payload['items'] ?? [];
  List<dynamic> get procurementItems => payload['items'] ?? [];
  String get goal => payload['goal'] ?? '';
  String get summary => payload['summary'] ?? '';
}
