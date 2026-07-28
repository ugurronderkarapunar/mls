def compare_groups(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    test_type: str = "auto",
) -> Tuple[str, float, float, str]:
    """İki veya daha fazla grup için uygun hipotez testi yapar.

    Args:
        df: DataFrame.
        group_col: Grupları ayıran kategorik sütun.
        value_col: Test yapılacak sayısal sütun.
        test_type: 'auto', 't-test', 'anova', 'mannwhitney', 'kruskal'.

    Returns:
        Tuple[str, float, float, str]: 
            - test_name (testin adı)
            - stat (test istatistiği)
            - p (p değeri)
            - result (yorum: "Anlamlı fark var" / "Anlamlı fark yok")
    """
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    if len(groups) < 2:
        return "En az 2 grup gerekli", np.nan, np.nan, ""

    # Otomatik test seçimi: normallik varsayımına göre
    if test_type == "auto":
        is_normal = True
        for g in groups:
            if len(g) >= 3:
                _, p_norm = stats.shapiro(g)
                if p_norm < 0.05:
                    is_normal = False
                    break
        if is_normal and len(groups) == 2:
            test_type = "t-test"
        elif is_normal and len(groups) > 2:
            test_type = "anova"
        elif len(groups) == 2:
            test_type = "mannwhitney"
        else:
            test_type = "kruskal"

    if test_type == "t-test":
        stat, p = stats.ttest_ind(groups[0], groups[1])
        test_name = "Bağımsız t-testi"
    elif test_type == "anova":
        stat, p = stats.f_oneway(*groups)
        test_name = "ANOVA"
    elif test_type == "mannwhitney":
        stat, p = stats.mannwhitneyu(groups[0], groups[1])
        test_name = "Mann-Whitney U"
    elif test_type == "kruskal":
        stat, p = stats.kruskal(*groups)
        test_name = "Kruskal-Wallis"
    else:
        return "Geçersiz test tipi", np.nan, np.nan, ""

    result = "Anlamlı fark var (p<0.05)" if p < 0.05 else "Anlamlı fark yok (p>=0.05)"
    return test_name, stat, p, result
