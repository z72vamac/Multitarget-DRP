import os
import win32com.client as win32

def avisador(iter, string):
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = 'cvalverde@us.es'
    ##Subject Line of Email
    if iter % 25 == 24:
        size = iter // 25 + 5
        mail.Subject = 'Solved instances of size {0} and {1}'.format(size, string)
        ##Body of Email
        mail.Body = 'Poquito a poco'
        
        mail.Send()

    # batch = []
    #
    # # iterating through everyfile in the folder
    # count = 0
    # for i in range(1, 11):
    #     count = count + 1
    #     #joining the OG file path plus the string of the filename
    #     # files = (os.path.join("C:\\Users\\myname\\Desktop\\InvoicesToSend",str(filename)))
    #     #attach the files
    #     # mail.Attachments.Add(files)
    #     if count == 10:
    #         #sending the mail after 10 attachments are added
    #         mail.Send()
    #         count = 0