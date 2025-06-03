import asyncio
import os
import queue
import shutil
import threading
from typing import List

import torch

from .global_vars import get_args


class CkptUploadQueue:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent reinitialization if already initialized.
        if hasattr(self, "_initialized") and self._initialized:
            return

        args = get_args()

        self.azcopy_command = "azcopy copy --output-level essential --log-level WARNING --recursive"
        if args.ckpt_upload_ingress_mbps > 0:
            if args.ckpt_format == "torch":
                self.azcopy_command += f" --cap-mbps {args.ckpt_upload_ingress_mbps}"
            else:
                nnodes = args.world_size // torch.cuda.device_count()
                self.azcopy_command += f" --cap-mbps {args.ckpt_upload_ingress_mbps // nnodes}"

        self.ckpt_iter_prefix = "iter_"
        self.ckpt_tracker_file = "latest_checkpointed_iteration.txt"
        self.log_progress_file = "progress.txt" if args.log_progress else None
        self.upload_metafiles = bool(args.rank == 0)
        self.local_dir = args.save
        self.blob_path = args.ckpt_upload_blob_path.rstrip("/")
        self.blob_sas_path = args.ckpt_upload_blob_sas_path
        self.blob_concurrency = args.ckpt_upload_blob_concurrency

        self.upload_tasks = queue.Queue()
        self._running = True

        # Start a background thread with its own asyncio event loop.
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        self._initialized = True

    def read_blob_sas_token(self) -> str:
        """
        Return an Azure Blob SAS token.

        If `blob_sas_path` exists and contains a non-empty token, read it and return.
        Otherwise, try the AZURE_SAS_TOKEN environment variable.
        """
        if os.path.isfile(self.blob_sas_path):
            with open(self.blob_sas_path, "r") as f:
                token = f.read().strip()
                if token:
                    print(f"Checkpoint blob SAS token: loaded from {self.blob_sas_path}")
                    return token
        print(f"Checkpoint blob SAS token: failed to load from {self.blob_sas_path}")

        token = os.getenv("AZURE_SAS_TOKEN")
        if token:
            print("Checkpoint blob SAS token: loaded from env AZURE_SAS_TOKEN")
            return token
        print("Checkpoint blob SAS token: failed to load from env AZURE_SAS_TOKEN")
        return ""

    def add_upload_task(self, upload_paths: List[str], on_success_delete: bool = True):
        """
        Enqueue a list of directories and files to be uploaded.
        """
        self.upload_tasks.put((upload_paths, on_success_delete))
        print("Checkpoint upload enqueued task: {} -> {}, {} delete on success".format(
            ", ".join(upload_paths),
            self.blob_path,
            "will" if on_success_delete else "will not",
        ))

    async def _run_task(self, upload_paths: List[str], on_success_delete: bool = True):
        """
        Run the checkpoint upload and deletion task asynchronously.
        """
        rc = await self._run_upload(upload_paths)
        if rc and self.upload_metafiles:
            metafile_paths = []
            try:
                uploaded_iter = max(
                    [int(path.split("_")[1]) if path.startswith(self.ckpt_iter_prefix) else 0 for path in upload_paths]
                )
                with open(os.path.join(self.local_dir, self.ckpt_tracker_file), "r") as f:
                    local_iter = int(f.read())
                if local_iter != uploaded_iter:
                    print(f"Local checkpoint {local_iter} is ahead of uploaded checkpoint {uploaded_iter}, consider to increase save interval")
                else:
                    metafile_paths.append(self.ckpt_tracker_file)
            except Exception as e:
                print(f"Failed to compare iteration between local and uploaded checkpoints due to: {e}")
            if self.log_progress_file:
                metafile_paths.append(self.log_progress_file)
            await self._run_upload(metafile_paths)
        if rc and on_success_delete:
            await self._run_delete(
                [path for path in upload_paths if path.startswith(self.ckpt_iter_prefix)]
            )

    async def _run_upload(self, upload_paths: List[str]) -> bool:
        """
        Run the azcopy command asynchronously to upload the file or directory.
        """
        if not upload_paths:
            return True
        include_path = ";".join(upload_paths)
        blob_url = f"{self.blob_path}?{self.read_blob_sas_token()}"
        command = f"{self.azcopy_command} '{self.local_dir}' '{blob_url}' --include-path '{include_path}'"
        print(f"Checkpoint upload started: {command}")
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"AZCOPY_CONCURRENCY_VALUE": self.blob_concurrency},
        )
        stdout, stderr = await proc.communicate()
        if stdout:
            print(f"Checkpoint upload stdout:{stdout.decode()}")
        if stderr:
            print(f"Checkpoint upload stderr:{stderr.decode()}")

        if proc.returncode == 0:
            print(f"Checkpoint upload succeeded for {include_path}.")
        else:
            print(f"Checkpoint upload failed for {include_path} with return code {proc.returncode}.")
        return bool(proc.returncode == 0)

    async def _run_delete(self, delete_paths: List[str]) -> bool:
        """
        Asynchronously delete the directories in local paths.
        """
        rc = True
        for path in delete_paths:
            abs_path = os.path.join(self.local_dir, path)
            if os.path.isdir(abs_path):
                try:
                    await asyncio.to_thread(shutil.rmtree, abs_path)
                    print(f"Checkpoint deleted: {abs_path}")
                except Exception as e:
                    rc = False
                    print(f"Checkpoint delete failed for {abs_path} due to: {e}")
        return rc

    def _worker(self):
        """
        Background worker thread that processes upload tasks.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            task = self.upload_tasks.get()
            if task is None:
                break
            loop.run_until_complete(self._run_task(*task))
            self.upload_tasks.task_done()
        loop.close()

    def stop(self):
        """
        Stop the worker thread gracefully.
        """
        print("Checkpoint upload is gracefully stopping")
        self._running = False
        self.upload_tasks.put(None)
        self.worker_thread.join()
        print("Checkpoint upload stopped")
