==========
Mask stats
==========

Compute summary statistics between two sets of N-dimensional binary masks.

Installation instructions
-------------------------

.. code::

    pip install mask_stats


Usage
-----

.. code:: python

        from mask_stats import compute_evaluations_for_mask_pairs

        true_masks = [mask1, mask2, ..., maskN]
        pred_masks = [pred1, pred2, ..., predN]

        true_eval, pred_eval = compute_evaluations_for_mask_pairs(true_masks, pred_masks)
