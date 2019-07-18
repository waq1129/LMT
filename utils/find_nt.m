function bb = find_nt(ll)

N = ll;
K = 1:ceil(N/2);
D = [K(rem(N,K)==0) N];

[~,aa] = sort(abs(D-1000));
bb = D(aa);

for ii=1:length(bb)
    bbii = bb(ii);
    if 1000-bbii>200
        continue;
    else
        break;
    end
end

bb = bbii;