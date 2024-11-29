import pandas as pd


# 기준 group 별 train, val, test 6:2:2 분리
def split_dataset(df, group, splits):
    if group == None:
        if splits == "val_test":
            train = df.sample(frac=0.5, random_state=42)
            test = df.drop(train.index)
            return train, test

        elif splits == "train_val_test":
            train = df.sample(frac=0.6, random_state=42)
            val_test = df.drop(train.index)
            val = val_test.sample(frac=0.5, random_state=42)
            test = val_test.drop(val.index)
            return train, val, test
    else:
        if splits == "val_test":
            train = df.groupby(group).sample(frac=0.5, random_state=42)
            test = df.drop(train.index)
            return train, test
        elif splits == "train_val_test":
            train = df.groupby(group).sample(frac=0.6, random_state=42)
            val_test = df.drop(train.index)
            val = val_test.groupby(group).sample(frac=0.5, random_state=42)
            test = val_test.drop(val.index)
            return train, val, test


# 1. Toxicity
def toxic_loader():
    train = pd.read_csv("./data/raw/Toxicity/ha_toxic_train.csv")
    train.drop(["label"], axis=1, inplace=True)
    train = train.loc[train["toxicity"] == 1]  # choose unsafe from train
    train.rename(columns={"toxicity": "label"}, inplace=True)
    train.reset_index(drop=True, inplace=True)

    val_test = pd.read_csv("./data/raw/Toxicity/ha_toxic_test.csv")
    val_test.drop(["Unnamed: 0"], axis=1, inplace=True)
    val_test.drop_duplicates(inplace=True)  # duplicates
    val_test.drop(["label"], axis=1, inplace=True)
    val_test.rename(columns={"toxicity": "label"}, inplace=True)
    val_test.reset_index(drop=True, inplace=True)

    dups = []  # duplicates
    for idx, i in enumerate(train["text"]):
        if len(val_test[val_test["text"] == i]) >= 1:
            dups.append(idx)
    train.drop(dups, inplace=True)

    val_test_unsafe = val_test.loc[val_test["label"] == 1]  # choose unsafe from test
    val_test_safe = val_test.loc[val_test["label"] == 0]  # choose safe from test
    val_test_safe = val_test_safe.sample(
        n=val_test_unsafe.shape[0], random_state=42
    )  # Balance safe and unsafe data count

    val_unsafe, test_unsafe = split_dataset(val_test_unsafe, None, "val_test")
    val_safe, test_safe = split_dataset(val_test_safe, None, "val_test")

    val = pd.concat([val_unsafe, val_safe])
    test = pd.concat([test_unsafe, test_safe])

    for dataset in [train, val, test]:
        dataset["dataset"] = "toxic"
        dataset["category"] = "toxic"
        dataset.reset_index(drop=True)

    return train, val, test


# 2. Misuse
def misuse_loader():
    ## safe
    safe = pd.read_json("./data/raw/Misuse/gpteacher_instruct.json")
    safe = safe.loc[safe["input"] == ""]  # 추가적인 input이 없는 것만 선택
    safe.rename(columns={"instruction": "text"}, inplace=True)
    safe["dataset"] = "gpteacher_instruct"
    safe["label"] = 0
    safe = safe[["text", "dataset", "label"]]

    ## unsafe
    unsafe = pd.read_json("./data/raw/Misuse/misuse_TrustLLM.json")
    unsafe.rename(columns={"prompt": "text", "source": "dataset"}, inplace=True)

    ## DAN
    unsafe_DAN = unsafe.loc[unsafe["dataset"] == "Do-anything-now"]
    unsafe_DAN["label"] = 1

    ## DNA
    unsafe_DNA = unsafe.loc[unsafe["dataset"] == "do_not_answer"]
    unsafe_DNA.drop(["type"], axis=1, inplace=True)
    unsafe_DNA[["first_class", "second_class", "third_class"]] = unsafe_DNA.label.apply(
        pd.Series
    )
    unsafe_DNA.drop(["label", "first_class", "second_class"], axis=1, inplace=True)
    unsafe_DNA.rename(columns={"third_class": "type"}, inplace=True)
    unsafe_DNA["label"] = 1

    ## ADD
    unsafe_ADD = unsafe.loc[unsafe["dataset"] == "misuse_add"]
    unsafe_ADD.drop(["type"], axis=1, inplace=True)

    unsafe_ADD.label = unsafe_ADD.label.apply(lambda x: x[0])
    unsafe_ADD.rename(columns={"label": "type"}, inplace=True)
    unsafe_ADD["label"] = 1

    unsafe_whole = pd.concat([unsafe_DAN, unsafe_DNA, unsafe_ADD])
    train, val_unsafe, test_unsafe = split_dataset(
        unsafe_whole, "type", "train_val_test"
    )

    val_test_safe = safe.sample(
        n=val_unsafe.shape[0] + test_unsafe.shape[0], random_state=42
    )
    val_safe, test_safe = split_dataset(val_test_safe, None, "val_test")
    val = pd.concat([val_unsafe, val_safe])
    test = pd.concat([test_unsafe, test_safe])

    for dataset in [train, val, test]:
        dataset["category"] = "misuse"
        dataset.reset_index(drop=True, inplace=True)

    return train, val, test


# 3. Jailbreaking
def jailbreaking_loader():

    ## safe
    safe = pd.read_json("./data/raw/Jailbreak/gpteacher_roleplay.json")
    safe = safe.loc[safe["input"] == ""]  # 추가적인 input이 없는 것만 선택
    safe.rename(columns={"instruction": "text"}, inplace=True)
    safe["dataset"] = "gpteacher_instruct"
    safe["label"] = 0
    safe = safe[["text", "dataset", "label"]]

    ## unsafe

    # IDW
    unsafe_ITW = pd.read_csv("./data/raw/Jailbreak/jailbreak_inthewild.csv")
    unsafe_ITW.rename(columns={"prompt": "text", "source": "type"}, inplace=True)
    unsafe_ITW["label"] = 1
    unsafe_ITW["dataset"] = "inthewild"
    unsafe_ITW = unsafe_ITW[["text", "dataset", "type", "label"]]
    unsafe_ITW.drop_duplicates(subset="text", inplace=True)
    unsafe_ITW.reset_index(drop=True, inplace=True)

    # Trust_Jailbreak
    unsafe_TJ = pd.read_json("./data/raw/Jailbreak/jailbreak_TrustLLM.json")
    unsafe_TJ.rename(
        columns={"prompt": "text", "label": "type", "source": "dataset"}, inplace=True
    )
    unsafe_TJ.type = unsafe_TJ.type.apply(lambda x: x[0])
    unsafe_TJ["label"] = 1

    unsafe_whole = pd.concat([unsafe_ITW, unsafe_TJ])
    unsafe_whole.reset_index(drop=True, inplace=True)
    train, val_unsafe, test_unsafe = split_dataset(
        unsafe_whole, "type", "train_val_test"
    )

    val_test_safe = safe.sample(
        n=val_unsafe.shape[0] + test_unsafe.shape[0], random_state=42
    )
    val_safe, test_safe = split_dataset(val_test_safe, None, "val_test")
    val = pd.concat([val_unsafe, val_safe])
    test = pd.concat([test_unsafe, test_safe])

    for dataset in [train, val, test]:
        dataset["category"] = "jailbreak"
        dataset.reset_index(drop=True, inplace=True)

    return train, val, test


# 4. exaggerated safety dataset
def exaggerated_safety_loader():
    xstest = pd.read_csv("./data/raw/Exaggerated_Safety/xstest_v2_prompts.csv")
    xstest.rename(columns={"toxicity": "label"}, inplace=True)
    xstest.drop(["id_v1", "id_v2", "focus", "note"], axis=1, inplace=True)
    xstest["dataset"] = "xstest"

    safe_xstest = xstest.loc[xstest["label"] == 0]  # choose safe from xstest
    unsafe_xstest = xstest.loc[xstest["label"] == 1]  # choose unsafe from xstest

    train, val_unsafe, test_unsafe = split_dataset(
        unsafe_xstest, "type", "train_val_test"
    )

    val_test_safe = safe_xstest.sample(
        n=val_unsafe.shape[0] + test_unsafe.shape[0], random_state=42
    )
    val_safe, test_safe = split_dataset(val_test_safe, None, "val_test")
    val = pd.concat([val_unsafe, val_safe])
    test = pd.concat([test_unsafe, test_safe])

    for dataset in [train, val, test]:
        dataset["category"] = "exaggerated"
        dataset.reset_index(drop=True, inplace=True)

    return train, val, test


def advbench_loader():
    advbench = pd.read_csv("./data/raw/advbench.csv").rename(columns={"goal": "text"})
    advbench["label"] = 1
    advbench["dataset"] = "advbench"
    advbench["category"] = "advbench"

    return advbench


def preprocessing():

    toxic_train, toxic_val, toxic_test = toxic_loader()
    misuse_train, misuse_val, misuse_test = misuse_loader()
    jailbreak_train, jailbreak_val, jailbreak_test = jailbreaking_loader()
    ex_train, ex_val, ex_test = exaggerated_safety_loader()
    advbench_test = advbench_loader()

    # toxic_train.to_csv("./data/toxic_train.csv")
    # toxic_val.to_csv("./data/toxic_val.csv")
    # toxic_test.to_csv("./data/toxic_test.csv")
    # misuse_train.to_csv("./data/misuse_train.csv")
    # misuse_val.to_csv("./data/misuse_val.csv")
    # misuse_test.to_csv("./data/misuse_test.csv")
    # jailbreak_train.to_csv("./data/jailbreak_train.csv")
    # jailbreak_val.to_csv("./data/jailbreak_val.csv")
    # jailbreak_test.to_csv("./data/jailbreak_test.csv")
    # ex_train.to_csv("./data/exaggerated_train.csv")
    # ex_val.to_csv("./data/exaggerated_val.csv")
    # ex_test.to_csv("./data/exaggerated_test.csv")

    # total train dataset
    train = pd.concat([toxic_train, misuse_train, jailbreak_train, ex_train])[
        ["text", "label", "category", "dataset"]
    ]
    train.reset_index(drop=True, inplace=True)
    train.to_csv("./data/train.csv", index=False)

    # total valid dataset
    val = pd.concat([toxic_val, misuse_val, jailbreak_val, ex_val])[
        ["text", "label", "category", "dataset"]
    ]
    val.reset_index(drop=True, inplace=True)
    val.to_csv("./data/val.csv", index=False)

    # total mix test dataset
    test = pd.concat([toxic_test, misuse_test, jailbreak_test, ex_test])[
        ["text", "label", "category", "dataset"]
    ]
    test.reset_index(drop=True, inplace=True)
    test.to_csv("./data/test.csv", index=False)

    advbench_test.to_csv("./data/adv_test.csv", index=False)

    print("Preprocessing Finish")

    return train, val, test
