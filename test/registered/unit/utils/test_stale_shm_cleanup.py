"""Tests for stale_shm_cleanup.py"""

import os
from unittest.mock import MagicMock, patch

from sglang.srt.utils.stale_shm_cleanup import (
    _creator_pid,
    _is_in_ci,
    _pid_alive,
    cleanup_stale_shm,
    make_shm_name,
)
from sglang.test.test_utils import CustomTestCase


class TestStaleShmCleanup(CustomTestCase):
    def test_make_shm_name(self):
        name = make_shm_name("test")
        self.assertTrue(name.startswith("sgl_shm_test_"))
        self.assertIn(str(os.getpid()), name)

    def test_creator_pid(self):
        self.assertEqual(_creator_pid(f"sgl_shm_kind_1234_abcd"), 1234)
        self.assertEqual(_creator_pid("multi_tokenizer_args_5678"), 5678)
        self.assertIsNone(_creator_pid("sgl_shm_invalid"))
        self.assertIsNone(_creator_pid("multi_tokenizer_args_notapid"))
        self.assertIsNone(_creator_pid(f"sgl_shm_kind_-1_abcd"))  # <=0

    @patch("sglang.srt.utils.stale_shm_cleanup.os.kill")
    def test_pid_alive(self, mock_kill):
        # Process exists
        mock_kill.return_value = None
        self.assertTrue(_pid_alive(1234))

        # Process does not exist
        mock_kill.side_effect = ProcessLookupError()
        self.assertFalse(_pid_alive(1234))

        # Process exists but permission denied
        mock_kill.side_effect = PermissionError()
        self.assertTrue(_pid_alive(1234))

    @patch.dict(os.environ, {"SGLANG_IS_IN_CI": "true"})
    def test_is_in_ci_true(self):
        self.assertTrue(_is_in_ci())

    @patch.dict(os.environ, {}, clear=True)
    def test_is_in_ci_false(self):
        self.assertFalse(_is_in_ci())

    @patch("sglang.srt.utils.stale_shm_cleanup._is_in_ci")
    @patch("sglang.srt.utils.stale_shm_cleanup._SHM_DIR")
    @patch("sglang.srt.utils.stale_shm_cleanup._pid_alive")
    @patch("sglang.srt.utils.stale_shm_cleanup.os.getpid")
    def test_cleanup_stale_shm_impl(
        self, mock_getpid, mock_pid_alive, mock_shm_dir, mock_is_in_ci
    ):
        mock_is_in_ci.return_value = True
        mock_shm_dir.is_dir.return_value = True
        mock_getpid.return_value = 9999

        # Setup mock files
        live_file = MagicMock()
        live_file.name = "sgl_shm_kind_1111_abcd"
        live_file.stat.return_value.st_size = 100

        dead_file = MagicMock()
        dead_file.name = "sgl_shm_kind_2222_abcd"
        dead_file.stat.return_value.st_size = 200

        own_file = MagicMock()
        own_file.name = "sgl_shm_kind_9999_abcd"

        non_sgl_file = MagicMock()
        non_sgl_file.name = "other_file"

        mock_shm_dir.iterdir.return_value = [
            live_file,
            dead_file,
            own_file,
            non_sgl_file,
        ]

        def mock_alive_impl(pid):
            return pid == 1111

        mock_pid_alive.side_effect = mock_alive_impl

        cleanup_stale_shm()

        # Only the dead file should be unlinked
        dead_file.unlink.assert_called_once()
        live_file.unlink.assert_not_called()
        own_file.unlink.assert_not_called()
        non_sgl_file.unlink.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
