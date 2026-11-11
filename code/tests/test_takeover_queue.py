import unittest
from takeover_queue import available, parse_npu


TABLE = '''
| 0     910B3               | OK            | 100.8       50                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          31133/ 65536         |
| 1     910B3               | OK            | 96.1        48                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          3200 / 65536         |
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
| 0       0                 | 3187429       | python                   | 27770                   |
'''


class QueueTests(unittest.TestCase):
    def test_busy_card_never_eligible_even_if_memory_fits(self):
        cards = parse_npu(TABLE)
        self.assertEqual(cards[0]['pids'], [3187429])
        self.assertFalse(available(cards[0], 10000))
        self.assertTrue(available(cards[1], 50000))

    def test_unknown_output_fails_closed(self):
        for text in ('', TABLE.split('Process id')[0], TABLE.replace('3200 / 65536', 'unknown')):
            with self.assertRaises(ValueError):
                parse_npu(text)

    def test_health_and_capacity_required(self):
        card = parse_npu(TABLE)[1]
        self.assertFalse(available({**card, 'health': 'Warning'}, 10000))
        self.assertFalse(available({**card, 'utilization': 100}, 10000))
        self.assertFalse(available(card, 64000))


if __name__ == '__main__':
    unittest.main()
