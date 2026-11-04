import copy
import unittest
from check_business_facts import make_case, normalized


class BusinessFactNormalizationTest(unittest.TestCase):
    def test_responsibility_prefix_and_action_order_do_not_change_facts(self):
        expected = make_case(7)['expected']
        parsed = copy.deepcopy(expected)
        parsed['actions'].reverse()
        for action in parsed['actions']:
            action['task'] = '负责' + action['task']
        self.assertEqual(normalized(parsed), normalized(expected))

    def test_wrong_owner_or_date_or_task_is_still_rejected(self):
        expected = make_case(7)['expected']
        for key, wrong in (('owner', '赵静'), ('due_date', '2026-11-29'), ('task', '取消接口联调')):
            with self.subTest(key=key):
                parsed = copy.deepcopy(expected)
                parsed['actions'][0][key] = wrong
                self.assertNotEqual(normalized(parsed), normalized(expected))

    def test_budget_and_pending_status_remain_strict(self):
        expected = make_case(7)['expected']
        for key, wrong in (('budget', 0), ('decision_status', '已批准')):
            parsed = copy.deepcopy(expected)
            parsed[key] = wrong
            self.assertNotEqual(normalized(parsed), normalized(expected))


if __name__ == '__main__':
    unittest.main()
