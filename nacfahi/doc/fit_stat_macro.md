Same as [`macro@fit!`], expect it computes [`FitStat`] instead of simple [`MinimizationReport`].

[`FitStat`] contains a bunch of other stuff you might want to know - namely, reduced $\chi^{2}$, a special kind of model reflecting parameter errors, and a covariance matrix.

Internally, invokes [`function@fit_stat`], see it's documentation for details.
