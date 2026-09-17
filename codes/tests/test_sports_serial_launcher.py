import unittest
from tools.run_sports_validation300 import PROFILES, verify_outcome


class SportsSerialLauncherTest(unittest.TestCase):
    def test_profiles_and_incomplete_success_exit(self):
        self.assertEqual(len(set(PROFILES)), 9)
        m = dict(status='validation_completed', student_config_profile=PROFILES[0],
                 final_test_performed=False, teacher_final_test_performed=False,
                 student_config_overrides={}, dataset_config_overrides={},
                 resolved_arguments=dict(epoch=300, early_stopping_patience=300,
                                         if_train_teacher=False))
        verify_outcome(m, PROFILES[0])
        for field, value in [('status','initialized'),('final_test_performed',True),
                             ('student_config_profile','wrong'),('student_config_overrides',{'seed':1})]:
            with self.assertRaises(RuntimeError):
                verify_outcome(dict(m, **{field:value}), PROFILES[0])


if __name__ == '__main__':
    unittest.main()
