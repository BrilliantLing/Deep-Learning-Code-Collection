Numbers = [4,14,24,33,45,54,64,74,88,98,108,118,128,134,149,157,165,178,189,195,203,210,216,221,232,244,246,255,263,276,288,297,305,317,320,337,342,353];
for i=1:length(Numbers)
    num = Numbers(i);
    if num<278
        delete(['0',num2str(num + 361),'.mat']);
        delete(['0',num2str(num + 722),'.mat']);
    else
        delete(['0',num2str(num + 361),'.mat']);
        delete([num2str(num + 722),'.mat']);
    end
    
end