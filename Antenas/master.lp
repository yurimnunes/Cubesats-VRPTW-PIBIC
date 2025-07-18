\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x5d2f7160f977ab99
Maximize
  x_DSS-24_0 + 4 x_DSS-24_1 + 3 x_DSS-24_2 + 4 x_DSS-24_3 + x_DSS-34_0
   + 4 x_DSS-34_1 + 3 x_DSS-34_2 + 4 x_DSS-34_3 + x_DSS-26_0 + 4 x_DSS-26_1
   + 3 x_DSS-26_2 + 4 x_DSS-26_3 + x_DSS-54_0 + 4 x_DSS-54_1 + 3 x_DSS-54_2
   + 4 x_DSS-54_3 + x_DSS-36_0 + 4 x_DSS-36_1 + 3 x_DSS-36_2
Subject To
 job_bfd97599-9ff3-4364-b5b8-5c7a9b34a4fc: x_DSS-24_0 + x_DSS-54_1
   + x_DSS-54_3 <= 1
 job_70ab4b76-ed81-40dd-ac90-4d280f5ff7c6: x_DSS-36_0 <= 1
 job_5f357cbe-214d-47bb-be53-5576a0995dd6: <= 1
 job_85e7b583-cb74-464f-9442-339256561000: <= 1
 job_73ab8546-9b71-4ca5-9846-cb5d64414ed3: <= 1
 job_3fb5fafd-bf2e-47e3-aaae-fea6bb857379: x_DSS-34_0 <= 1
 job_4390c648-c354-4a25-bc5d-5a598d2adcb0: x_DSS-26_0 <= 1
 job_30ccd467-154c-4a9a-9c44-134ada8ef05d: <= 1
 job_cd9c34fb-3b53-4e6f-a490-4c417721b09f: x_DSS-26_1 <= 1
 job_d3784dfb-8fc5-4545-a65a-0113372da6aa: <= 1
 job_ff377b8d-70af-4fcd-af1a-31308b063962: x_DSS-54_0 <= 1
 job_53d22d3a-93eb-4c84-915a-f13610d0fe30: <= 1
 job_8e295f03-10bb-4da1-9d2b-6d992c3b9c40: x_DSS-24_1 + x_DSS-24_2
   + x_DSS-24_3 + x_DSS-34_1 + x_DSS-34_2 + x_DSS-34_3 + x_DSS-26_1
   + x_DSS-26_2 + x_DSS-26_3 <= 1
 job_7eabc4fd-23b7-4eed-883c-b6499e751f52: x_DSS-36_1 + x_DSS-36_2 <= 1
 job_477b84bd-228c-41f2-a73c-80b51f678277: <= 1
 job_72b16e5c-01e5-48c8-a050-1b51ed750f72: x_DSS-54_1 + x_DSS-54_2
   + x_DSS-36_1 <= 1
 job_4f706cd7-ccb0-485f-b2af-0c29873d3592: x_DSS-24_1 + x_DSS-34_1 <= 1
 job_840c2e89-3905-44cf-ad5a-b3a0303d2e98: <= 1
 job_a8000b32-fa3b-47e5-91bc-6a3087c9aa6c: <= 1
 job_48d43566-0c1e-4eb4-9fd1-2e7cce346588: <= 1
 antenna_DSS-24: x_DSS-24_0 + x_DSS-24_1 + x_DSS-24_2 + x_DSS-24_3 <= 1
 antenna_DSS-34: x_DSS-34_0 + x_DSS-34_1 + x_DSS-34_2 + x_DSS-34_3 <= 1
 antenna_DSS-26: x_DSS-26_0 + x_DSS-26_1 + x_DSS-26_2 + x_DSS-26_3 <= 1
 antenna_DSS-54: x_DSS-54_0 + x_DSS-54_1 + x_DSS-54_2 + x_DSS-54_3 <= 1
 antenna_DSS-36: x_DSS-36_0 + x_DSS-36_1 + x_DSS-36_2 <= 1
Bounds
End
