function x=new_zigzag(A)
b=1;
for a=1:8
    if mod(a,2)==1
        for j=1:a
            i=a+1-j;
            x(1,b)=A(i,j);
            b=b+1;
        end
    else 
        for i=1:a
            j=a+1-i;
            x(1,b)=A(i,j);
            b=b+1;
        end
    end
end
for a=[7,6,5,4,3,2,1]
    if mod(a,2)==1
        for j=9-a:8;
            i=17-a-j;
            x(1,b)=A(i,j);
            b=b+1;
        end
    else
        for i=9-a:8
            j=17-a-i;
            x(1,b)=A(i,j);
            b=b+1;
        end
    end
end
end
