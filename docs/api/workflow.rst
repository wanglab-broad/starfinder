starfinder.workflow
===================

The engine-neutral task layer of the workflow: the plan, which validates a
configuration in the Python shape and writes the resolved configuration and the
manifest of every task, and the completion and staleness rule. The tasks run
through the console script, ``starfinder workflow <command> --manifest <path>
--task <id>``, one command per API entry. The
:doc:`migration guide <../migration>` lists the configuration keys that moved.

.. currentmodule:: starfinder.workflow

Plan and verify
---------------

.. autosummary::
   :toctree: generated

   plan
   verify

Refusals
--------

.. autosummary::
   :toctree: generated

   PlanRefused
   ResumeRefused
   TaskRefused

Task kinds
----------

.. autosummary::
   :toctree: generated

   TASK_BLOCKS
