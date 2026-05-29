import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_frontend/models/ai_event.dart';

void main() {
  test('AIEvent parses persisted backend event payloads', () {
    final event = AIEvent.fromJson({
      'type': 'ai_event',
      'event_type': 'procurement_plan',
      'narrative': 'Generated a procurement plan.',
      'data': {
        'goal': 'Dinner service',
        'items': [
          {'name': 'Tomatoes', 'quantity': '12'}
        ],
      },
    });

    expect(event.eventType, 'procurement_plan');
    expect(event.isProcurementEvent, isTrue);
    expect(event.narrative, 'Generated a procurement plan.');
    expect(event.goal, 'Dinner service');
    expect(event.procurementItems, hasLength(1));
  });
}
