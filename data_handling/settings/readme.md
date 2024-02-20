This folder contains settings for the input data. Provided settings are:

- Features: Column index starting from 0 (type=0 for continuous, 1 for categorical). Currently only supports continuous features!
- Label column: The column of the labels. This is not necessary for the application, but for our experimental settings. Can be None if to be ignored.
- Background label: Give the label for the background. If label is not background then it is anomalous.
- Has header: If file has header then we discard the first row.
- 