using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public interface IProductionLayer
    {
        Tensor Forward(Tensor data);
    }
}
